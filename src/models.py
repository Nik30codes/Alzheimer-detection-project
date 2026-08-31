"""Model definitions: custom CNN (Phase C) and MobileNetV2 / EfficientNet-B0 transfer
learning (Phase D)."""

import torch.nn as nn
import torchvision.models as tvm


def conv_block(in_ch, out_ch):
    """Conv -> BatchNorm -> ReLU, twice. BatchNorm is what makes this trainable
    at a decent learning rate without it being fiddly -- keeps activations in a
    stable range as they pass through the network."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """
    A compact from-scratch CNN, built from four conv_blocks (32->64->128->256
    channels) each followed by a 2x2 max-pool, then Global Average Pooling
    instead of Flatten before the classifier head.

    Why GlobalAveragePooling instead of Flatten+Dense (which the user's earlier
    from-scratch script used): Flatten keeps every spatial position as a separate
    weight, which on a 224x224 input balloons the classifier into millions of
    extra parameters that mostly just memorize the training set. GAP collapses
    each feature map to a single average value first, which is both far more
    parameter-efficient and less prone to overfitting on ~10k training images.
    It's also the same pattern MobileNetV2/EfficientNet use, so Phase D's models
    stay directly comparable to this one.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        """in_channels=1 is the original single-slice setup. in_channels=3 lets the
        same architecture take the 2.5D three-adjacent-slice stack from
        datasets.MRI25DDataset, so the 2.5D comparison covers this model too and not
        just the torchvision ones."""
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32), nn.MaxPool2d(2),     # 224 -> 112
            conv_block(32, 64), nn.MaxPool2d(2),     # 112 -> 56
            conv_block(64, 128), nn.MaxPool2d(2),    # 56 -> 28
            conv_block(128, 256), nn.MaxPool2d(2),   # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # (batch, 256, 14, 14) -> (batch, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def conv_block_3d(in_ch, out_ch):
    """Conv3d -> BatchNorm3d -> ReLU, twice. 3D analogue of conv_block."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class Simple3DCNN(nn.Module):
    """
    A true 3D CNN over a subject's whole 32-slice axial stack at once, instead of
    scoring each slice independently and averaging (what every other model in this
    project does). Motivation: decision 30 (slice_attention.py) measured the 32
    per-slice predictions as only ~1.3 effective independent measurements after
    averaging -- a 2D-then-average model structurally cannot recover cross-slice
    information that never made it into any single slice's prediction. A Conv3d
    stack sees all 32 slices jointly from the first layer on.

    Channel widths (16->32->64->128) are HALF of SimpleCNN's (32->64->128->256), not
    an arbitrary shrink: a Conv3d kernel has ~3x the parameters of the equivalent
    Conv2d at the same channel width (27 taps vs 9), and the AD/CN pool this trains on
    (~350 subjects per CV fold) is smaller than what would justify a naive 2D->3D
    parameter-count port. ~0.9M params, comparable to or smaller than SimpleCNN's
    1.2M despite operating on 32x the input pixels.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            conv_block_3d(in_channels, 16), nn.MaxPool3d(2),   # (32,224,224) -> (16,112,112)
            conv_block_3d(16, 32), nn.MaxPool3d(2),            # -> (8,56,56)
            conv_block_3d(32, 64), nn.MaxPool3d(2),            # -> (4,28,28)
            conv_block_3d(64, 128), nn.MaxPool3d(2),           # -> (2,14,14)
        )
        self.pool = nn.AdaptiveAvgPool3d(1)  # (batch, 128, 2, 14, 14) -> (batch, 128, 1, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_mobilenetv2(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """MobileNetV2, with its 1000-class head replaced by ours. Expects 3-channel input
    (our grayscale slices get replicated to 3 channels in datasets.py) with ImageNet
    normalization, not the 1-channel input SimpleCNN uses.

    pretrained=False loads random initial weights instead of ImageNet ones -- used to
    test whether the pretrained weights are actually helping on this dataset (grayscale
    MRI is a large domain shift from ImageNet photos) or whether they're a net negative
    ("negative transfer"), by comparing against the same architecture trained from scratch."""
    weights = tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tvm.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """Same idea as build_mobilenetv2, for EfficientNet-B0.

    Note the default here is pretrained=False, unlike build_mobilenetv2's True. That's
    deliberate: MobileNetV2 fine-tuned from ImageNet weights was tried three ways on this
    dataset and every variant landed at 34.8-40.9% subject-level accuracy, well below the
    same architecture trained from random init (54.5%) and below the from-scratch custom
    CNN (56.1%). Grayscale MRI duplicated into fake-RGB channels is far enough from
    ImageNet photos that the pretrained features are a net negative here. Starting
    EfficientNet-B0 from scratch avoids repeating that detour."""
    weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module):
    """Phase 1 of fine-tuning: freeze every pretrained weight, leave only the new
    classifier head trainable. Trains fast and can't wreck the pretrained features."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_top_layers(model: nn.Module, fraction: float = 0.3):
    """
    Phase 2 of fine-tuning, done conservatively for a small dataset (~300 subjects):
    unfreeze only the LAST `fraction` of the backbone's blocks, plus the classifier.
    Early layers stay frozen.

    Why not unfreeze everything (what we tried first): early layers in an ImageNet-
    pretrained network learn generic, low-level features -- edges, textures, simple
    shapes -- that are broadly useful and don't need to change for MRI. Unfreezing
    them anyway, with only a few hundred training subjects, let the model overwrite
    that useful pretrained knowledge with dataset-specific noise: train accuracy kept
    climbing (63%->74%) while validation accuracy plateaued and test accuracy came in
    even lower -- a textbook overfitting signature. Later layers learn more
    task-specific patterns, which is what actually benefits from adapting to MRI.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    blocks = list(model.features.children())
    n_unfrozen = max(1, round(len(blocks) * fraction))
    for block in blocks[-n_unfrozen:]:
        for param in block.parameters():
            param.requires_grad = True


def freeze_batchnorm(model: nn.Module):
    """
    Keeps every BatchNorm layer's running mean/var fixed (eval-mode behavior) and its
    scale/shift weights frozen, even while surrounding conv layers are being trained.

    Why this matters here specifically: BatchNorm normalizes using statistics computed
    from the CURRENT batch during training. Our batches (32 images) are small and drawn
    from a small dataset (~300 subjects) that looks nothing like the millions of ImageNet
    photos these statistics were originally computed from. Letting BatchNorm recompute
    its statistics on our data mid-fine-tune adds another way for the model to overfit,
    on top of the weights themselves -- freezing it removes that source of instability.
    Standard practice when fine-tuning pretrained CNNs on small datasets.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
