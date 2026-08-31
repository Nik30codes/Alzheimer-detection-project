"""Image decoding, validation, preprocessing and inference.

Pure functions with no web framework in them, so the HTTP layer stays thin and this
logic is testable on its own.
"""
import base64
import io
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling modules
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datasets import EVAL_TRANSFORM_RGB, EVAL_TRANSFORM              # noqa: E402
from data_prep import BOTTLENECK_SIZE, OUT_SIZE                      # noqa: E402
from models import SimpleCNN, build_mobilenetv2, build_efficientnet_b0  # noqa: E402
from gradcam import GradCAM                                          # noqa: E402
from tasks import get_task                                           # noqa: E402

# Default to CPU, and only use CUDA when explicitly asked for via MMAD_DEVICE=cuda.
#
# Two reasons. (1) Deployment is CPU-only (SRS NFR-4), so CPU is the configuration that
# actually ships and therefore the one worth exercising locally. (2) This machine has a
# 6GB card that training jobs fill; a webapp holding a CUDA context alongside a training
# run pushed the GPU to 94% occupancy and slowed one cross-validation fold from an
# expected ~60 minutes to 569. Single-slice inference takes 0.05-1.2s on CPU, so there
# is nothing to gain from the GPU here.
DEVICE = torch.device(
    "cuda" if os.environ.get("MMAD_DEVICE", "cpu").lower() == "cuda"
    and torch.cuda.is_available() else "cpu")
_LOADED = {}


class InvalidInput(Exception):
    """Raised when the upload is not usable. Message is shown to the user verbatim."""


def _build(arch, n_out):
    if arch == "custom_cnn":
        return SimpleCNN(num_classes=n_out, in_channels=1), False
    if arch == "mobilenetv2":
        return build_mobilenetv2(n_out, pretrained=False), True
    if arch == "efficientnet_b0":
        return build_efficientnet_b0(n_out, pretrained=False), True
    raise ValueError(arch)


def load_task_model(task_id):
    """Build and cache one task's model. Refuses leaky checkpoints outright."""
    if task_id in _LOADED:
        return _LOADED[task_id]
    task = get_task(task_id)
    if task is None:
        raise KeyError(task_id)
    if "LEAKY" in task["checkpoint"].upper():
        # Those scored ~96% only on people already in the training set and fall to
        # roughly chance on a stranger's scan; serving one would confidently
        # misdiagnose every real user.
        raise ValueError("refusing to serve a leaky checkpoint")

    path = os.path.join(ROOT, "models", "checkpoints", task["checkpoint"])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model, rgb = _build(task["arch"], len(task["classes"]))
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    target = model.features[-1] if hasattr(model, "features") else None
    _LOADED[task_id] = {"model": model, "rgb": rgb,
                        "cam": GradCAM(model, target) if target is not None else None}
    return _LOADED[task_id]


def dicom_plane(ds):
    """Acquisition plane from ImageOrientationPatient, or None if the tag is absent.

    The slice normal is the cross product of the row and column direction cosines;
    whichever patient axis it points along names the plane. This is the only reliable
    way to know orientation -- validate_slice() cannot tell a sagittal head from an
    axial one, because both sit centred on a dark background.
    """
    iop = ds.get("ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return None
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return ["SAGITTAL", "CORONAL", "AXIAL"][int(np.argmax(np.abs(n)))]


def decode_image(file_bytes, filename=""):
    """Decode PNG/JPEG, or DICOM via pydicom. Returns (uint8 grayscale, kind).

    DICOM pixel data is scanner-native intensity, often 12-bit, not 0-255. It is
    rescaled with the same 0.5/99.5 percentile clip the training pipeline used so an
    uploaded DICOM lands in the intensity range the model was trained on.

    The acquisition plane is checked here, because only DICOM states it. ADNI T1 series
    are acquired sagittally and this model reads axial slices; without this check a
    sagittal slice would pass the geometric tests and get a confident, meaningless
    answer.
    """
    if not filename.lower().endswith(".dcm"):
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img, "image"
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        px = ds.pixel_array.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        raise InvalidInput(f"Could not read that file as an image or DICOM ({e}).")

    plane = dicom_plane(ds)
    if plane is not None and plane != "AXIAL":
        raise InvalidInput(
            f"This DICOM was acquired in the {plane.lower()} plane, but the model reads "
            "AXIAL (top-down) slices. ADNI's T1 scans are stored sagittally, so a raw "
            "ADNI file will hit this message. Reconstruct an axial slice first, or "
            "upload an axially-acquired image.")

    if px.ndim == 3:                       # multi-frame: take the middle frame
        px = px[px.shape[0] // 2]
    lo, hi = np.percentile(px, [0.5, 99.5])
    px = np.clip(px, lo, hi)
    return ((px - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8), "dicom"


def validate_slice(img):
    """Cheap geometric checks for 'is this an axial brain slice'.

    Not a classifier -- these are properties every head-on-black-background scan has,
    chosen to catch the common failure mode: a photograph returning a confident
    Alzheimer's probability. Raises InvalidInput with a human-readable reason.
    """
    h, w = img.shape[:2]
    if min(h, w) < 64:
        raise InvalidInput("Image is too small to be an MRI slice (minimum 64x64 pixels).")
    if max(h, w) / min(h, w) > 2.0:
        raise InvalidInput("This image is far from square. Axial MRI slices are roughly "
                           "square — this looks like a photograph or screenshot.")
    f = img.astype(np.float32) / 255.0
    k = max(4, min(h, w) // 10)
    corners = np.concatenate([f[:k, :k].ravel(), f[:k, -k:].ravel(),
                              f[-k:, :k].ravel(), f[-k:, -k:].ravel()])
    if corners.mean() > 0.35:
        raise InvalidInput("The corners of this image are not dark. An axial MRI slice "
                           "shows the head centred on a black background.")
    fg = float((f > 0.15).mean())
    if not 0.10 < fg < 0.85:
        raise InvalidInput(f"Bright pixels cover {fg:.0%} of the frame, which does not "
                           "look like a head on a dark background.")
    ys, xs = np.nonzero(f > 0.15)
    if len(xs) == 0:
        raise InvalidInput("This image appears to be blank.")
    if abs(xs.mean() / w - 0.5) > 0.22 or abs(ys.mean() / h - 0.5) > 0.22:
        raise InvalidInput("The bright region is off-centre. In an axial MRI the head "
                           "sits in the middle of the frame. If this is a sagittal "
                           "(side-on) slice it will not work — this model reads axial.")


def harmonise(img):
    """Bring an arbitrary upload to the resolution the model was trained on.

    Training slices passed through a common 144px stage to equalise two ADNI protocols
    with different native resolutions, then up to 224px. An unknown upload has to go
    through the same funnel or the model sees detail it never saw in training.

    BUT an image that is ALREADY exactly OUT_SIZE has, in practice, already been through
    that pipeline -- every slice this project exports is a 224x224 PNG produced by
    data_prep. Re-applying the bottleneck low-passes it a second time and costs real
    accuracy: measured over all 75 AD-vs-CN test subjects, double-harmonising plus
    mirrored TTA scored 81.3% while the training-time path scored 82.67%, matching the
    recorded per-subject probabilities to a mean absolute difference of 0.0003.

    So: funnel anything that is not already at OUT_SIZE, and pass through anything that
    is. The pass-through case is exactly the case where the funnel is redundant.
    """
    if img.shape[0] == OUT_SIZE and img.shape[1] == OUT_SIZE:
        return img
    img = cv2.resize(img, (BOTTLENECK_SIZE, BOTTLENECK_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)


def to_data_uri(im):
    ok, buf = cv2.imencode(".png", im)
    return "data:image/png;base64," + base64.b64encode(buf).decode() if ok else None


def predict(img, task_id, tta=False):
    """Score one slice. Returns (probs, state, task, input_tensor).

    tta=False by default so this reproduces the published estimator EXACTLY. The
    reported accuracy and ROC AUC come from `scripts/train_binary_adni1.py`, whose soft
    vote is a plain mean of un-augmented per-slice softmax -- it does not mirror. Adding
    mirrored TTA here made the demo disagree with its own headline figure (81.3% vs
    82.67% over the 75 test subjects). A demo that cannot reproduce the number printed
    above it is worse than one that skips a mild augmentation.
    """
    task = get_task(task_id)
    state = load_task_model(task_id)
    transform = EVAL_TRANSFORM_RGB if state["rgb"] else EVAL_TRANSFORM
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(state["model"](x), dim=1)
        if tta:
            p = (p + torch.softmax(state["model"](torch.flip(x, dims=[3])), dim=1)) / 2
        probs = p[0].cpu().numpy()
    return probs, state, task, x


def gradcam_overlay(state, x, img, class_idx):
    if state["cam"] is None:
        return None
    cam, _, _ = state["cam"](x, class_idx=int(class_idx))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return to_data_uri((0.55 * base + 0.45 * heat).astype(np.uint8))
