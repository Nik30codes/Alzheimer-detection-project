"""Real-model regression checks. Run from the repository root:

python -B -m unittest discover -s webapp/tests -v
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
from fastapi import HTTPException, UploadFile
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))
import inference  # noqa: E402
import main  # noqa: E402

torch.set_num_threads(2)


def analyse(files):
    return [json.loads(event) for event in main._analyse(files, "ad_vs_cn")][-1]


def make_dicom(pixels, orientation=True, photo="MONOCHROME2"):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photo
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if pixels.ndim == 3:
        ds.NumberOfFrames = pixels.shape[0]
    if orientation:
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


class InferenceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples = ROOT / "webapp/frontend/public/samples"
        cls.cn = (samples / "cn_1.png").read_bytes()
        cls.ad = (samples / "ad_1.png").read_bytes()

    def test_concurrent_predictions_and_heatmaps_match_sequential(self):
        inputs = [(self.cn, "cn.png"), (self.ad, "ad.png")]
        reference = [analyse([item])["result"] for item in inputs]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: analyse([inputs[i % 2]]), range(16)))
        for i, event in enumerate(results):
            self.assertIn("result", event)
            self.assertEqual(event["result"]["prediction"], reference[i % 2]["prediction"])
            self.assertEqual(event["result"]["gradcam"], reference[i % 2]["gradcam"])

    def test_mixed_subjects_rejected(self):
        event = analyse([(self.cn, "001_S_0001_001.png"),
                         (self.ad, "002_S_0002_001.png")])
        self.assertIn("more than one subject", event["error"])
        self.assertNotIn("result", event)

    def test_partial_upload_reports_single_accepted_slice(self):
        result = analyse([(self.cn, "cn.png"), (b"invalid", "bad.png")])["result"]
        self.assertEqual(result["n_slices_used"], 1)
        self.assertEqual(result["n_slices_skipped"], 1)
        self.assertFalse(result["subject_level"])
        self.assertIn("Single slice", result["aggregation"])

    def test_unknown_identity_is_explicit(self):
        result = analyse([(self.cn, "first.png"), (self.cn, "second.png")])["result"]
        self.assertFalse(result["subject_identity_verified"])
        self.assertIn("cannot verify", result["input_note"])

    def test_uniform_shape_rejected(self):
        image = np.zeros((224, 224), np.uint8)
        cv2.circle(image, (112, 112), 75, 180, -1)
        with self.assertRaisesRegex(inference.InvalidInput, "intensity variation"):
            inference.validate_slice(image)

    def test_dicom_orientation_required(self):
        image = np.arange(224 * 224, dtype=np.uint16).reshape(224, 224)
        with self.assertRaisesRegex(inference.InvalidInput, "orientation"):
            inference.decode_image(make_dicom(image, orientation=False), "scan.dcm")

    def test_multiframe_not_silently_reduced(self):
        with self.assertRaisesRegex(inference.InvalidInput, "Multi-frame"):
            inference.decode_image(make_dicom(np.zeros((3, 224, 224), np.uint16)), "scan.dcm")

    def test_monochrome1_matches_equivalent_monochrome2(self):
        image = np.arange(224 * 224, dtype=np.uint16).reshape(224, 224)
        expected, _ = inference.decode_image(make_dicom(image), "scan.dcm")
        actual, _ = inference.decode_image(
            make_dicom(image.max() - image, photo="MONOCHROME1"), "scan.dcm")
        np.testing.assert_array_equal(actual, expected)

    def test_upload_handles_closed_on_success_and_rejection(self):
        for data, task, status in [(self.cn, "ad_vs_cn", None),
                                   (self.cn, "unknown", 400),
                                   (b"0" * (main.MAX_BYTES + 1), "ad_vs_cn", 413)]:
            stream = tempfile.SpooledTemporaryFile(max_size=main.MAX_BYTES + 2)
            stream.write(data)
            stream.seek(0)
            upload = UploadFile(filename="test.png", file=stream)
            if status:
                with self.assertRaises(HTTPException) as caught:
                    asyncio.run(main.predict_stream([upload], task))
                self.assertEqual(caught.exception.status_code, status)
            else:
                asyncio.run(main.predict_stream([upload], task))
            self.assertTrue(upload.file.closed)


if __name__ == "__main__":
    unittest.main()
