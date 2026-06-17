"""Unified multi-task model."""

from pathlib import Path

import torch
import torch.nn as nn

from .classification import ClassificationHead
from .localization import BoundingBoxHead
from .segmentation import UNetDecoder
from .vgg11 import VGG11Encoder
from utils import initialize_multitask_from_task_checkpoints
CLASSIFIER_DRIVE_ID = "1xeZC_3GK7Fg-v71Pf5D2eA__opz91hVT"
LOCALIZER_DRIVE_ID = "1YZzeFPW0E0vfDYFFOUa82xIqDMa3ZL4p"
UNET_DRIVE_ID = "1AOHOclD5cskBlhLXmgB-79YT4nuFA_A0"

def _download_required_checkpoints() -> tuple[Path, Path, Path]:
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = checkpoint_dir / "classifier.pth"
    localizer_path = checkpoint_dir / "localizer.pth"
    unet_path = checkpoint_dir / "unet.pth"

    downloads = (
        (CLASSIFIER_DRIVE_ID, classifier_path),
        (LOCALIZER_DRIVE_ID, localizer_path),
        (UNET_DRIVE_ID, unet_path),
    )
    valid_downloads = [(file_id, output_path) for file_id, output_path in downloads if file_id and not file_id.startswith("REPLACE_WITH_")]
    if not valid_downloads:
        return classifier_path, localizer_path, unet_path

    try:
        import gdown
    except Exception:
        # Keep initialization resilient in restricted/no-network environments.
        return classifier_path, localizer_path, unet_path

    for file_id, output_path in valid_downloads:
        if output_path.exists():
            continue
        try:
            # Lazily populate missing checkpoints so eval scripts can run with no manual setup.
            gdown.download(id=file_id, output=str(output_path), quiet=False)
        except Exception as exc:
            # Do not crash model construction if one checkpoint URL is unavailable.
            print(f"Warning: failed to download checkpoint {output_path.name}: {exc}")
    return classifier_path, localizer_path, unet_path


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        classifier_path, localizer_path, unet_path = _download_required_checkpoints()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.output_channels,
            num_classes=num_breeds,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
        )
        self.localization_head = BoundingBoxHead(in_channels=self.encoder.output_channels)
        self.segmentation_head = UNetDecoder(num_classes=seg_classes, use_batchnorm=use_batchnorm)
        initialize_multitask_from_task_checkpoints(
            self,
            classifier_checkpoint=classifier_path if classifier_path.exists() else None,
            localizer_checkpoint=localizer_path if localizer_path.exists() else None,
            segmentation_checkpoint=unet_path if unet_path.exists() else None,
            device="cpu",
        )

    def forward(self, x: torch.Tensor):
        """Return outputs for all three tasks in a single forward pass."""
        bottleneck, features = self.encoder(x, return_features=True)
        bbox_norm = self.localization_head(bottleneck)
        h, w = x.shape[-2], x.shape[-1]
        # Keep localization head normalized for training stability and convert to image-space here.
        scale = torch.tensor([w, h, w, h], dtype=bbox_norm.dtype, device=bbox_norm.device).unsqueeze(0)
        bbox_xywh = bbox_norm * scale
        return {
            "classification": self.classification_head(bottleneck),
            "localization": bbox_xywh,
            "segmentation": self.segmentation_head(bottleneck, features),
        }
