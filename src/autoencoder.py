"""Self-supervised masked autoencoder for MRI slices.

Purpose: the four-way task has only 432 labelled training subjects, and every
architecture overfits within ~10 epochs. Labels are the scarce resource, not images
-- there are 27,296 extracted slices. A masked autoencoder learns from the images
alone, with no labels at all: parts of the slice are blanked out and the network must
rebuild them, which forces it to learn what a brain actually looks like. The encoder
is then reused as the starting point for the classifier.

Why this should work here when ImageNet pretraining did not (decision 7): the failure
there was the SOURCE DOMAIN. ImageNet is natural RGB photographs, and every
fine-tuning strategy scored below a from-scratch model. Here the pretraining images
ARE this dataset's own MRI slices -- same modality, same anatomy, same slice
positions, same preprocessing. Published work supports this: brain-specific
self-supervised pretraining outperforms both general medical and natural-image
pretraining, and matches supervised performance using a fraction of the labels.

The encoder is deliberately identical to SimpleCNN.features, so its weights load
straight into the classifier with no adapter layer.

LEAKAGE NOTE: pretraining uses TRAIN-SPLIT SUBJECTS ONLY. Reconstructing validation
or test images is unsupervised, but it is still fitting the model to data that later
gets used for evaluation, and this project treats that as leakage.
"""

import torch
import torch.nn as nn

from models import conv_block


def up_block(in_ch, out_ch):
    """Nearest-neighbour upsample then conv, rather than ConvTranspose2d.

    Transposed convolutions produce the familiar checkerboard artefacts, which on a
    reconstruction task the encoder would have to spend capacity compensating for --
    capacity that should be going into representing anatomy.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Encoder mirrors SimpleCNN.features exactly (224 -> 14, 1 -> 256 channels);
    decoder walks back up to a single-channel 224x224 reconstruction.

    The output uses tanh because the input transform is Normalize(0.5, 0.5), which
    puts pixels in [-1, 1]. A sigmoid would cap at [0, 1] and make half the range
    unreachable.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32), nn.MaxPool2d(2),   # 224 -> 112
            conv_block(32, 64), nn.MaxPool2d(2),            # 112 -> 56
            conv_block(64, 128), nn.MaxPool2d(2),           # 56  -> 28
            conv_block(128, 256), nn.MaxPool2d(2),          # 28  -> 14
        )
        self.decoder = nn.Sequential(
            up_block(256, 128),   # 14  -> 28
            up_block(128, 64),    # 28  -> 56
            up_block(64, 32),     # 56  -> 112
            up_block(32, 32),     # 112 -> 224
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.features(x))

    def encoder_state_dict(self):
        """Weights keyed so they drop straight into SimpleCNN (which also calls its
        feature extractor `features`)."""
        return {f"features.{k}": v for k, v in self.features.state_dict().items()}


def random_mask(x, patch=32, n_patches=8):
    """Blank out n_patches random squares per image.

    Masking rather than plain reconstruction matters: an unmasked autoencoder can
    score well by learning an identity-like shortcut through the bottleneck, which
    teaches it little. Forcing it to inpaint missing regions means it has to model
    how brain structures relate to their surroundings -- for example inferring a
    ventricle's shape from the tissue around it -- which is the kind of structural
    understanding the classifier needs.

    Returns (masked_input, mask) where mask is 1 on the blanked pixels.
    """
    b, _, h, w = x.shape
    mask = torch.zeros(b, 1, h, w, device=x.device)
    for _ in range(n_patches):
        ys = torch.randint(0, h - patch, (b,), device=x.device)
        xs = torch.randint(0, w - patch, (b,), device=x.device)
        for i in range(b):
            mask[i, :, ys[i]:ys[i] + patch, xs[i]:xs[i] + patch] = 1.0
    return x * (1 - mask), mask
