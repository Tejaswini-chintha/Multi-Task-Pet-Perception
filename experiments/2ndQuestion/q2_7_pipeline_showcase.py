"""Question 2.7: run the final pipeline on novel pet images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import load_model, load_single_image, make_wandb_image_from_bbox, overlay_mask, wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def mask_to_rgb(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    color_map = np.array(
        [
            [255, 0, 0],    # class 0
            [0, 255, 0],    # class 1
            [0, 0, 255],    # class 2
        ],
        dtype=np.uint8,
    )
    return color_map[np.clip(mask, 0, 2)]


def overlay_to_uint8(image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> np.ndarray:
    overlay = np.clip(overlay_mask(image_tensor, mask_tensor), 0.0, 1.0)
    return (overlay * 255.0).astype(np.uint8)


def main():
    args = parse_args()
    model = load_model("multitask", args.checkpoint)
    device = next(model.parameters()).device
    image_paths = sorted([p for p in Path(args.images_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])[:3]

    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_7_pipeline_showcase")
        log_payload = {}
        for image_path in image_paths:
            image = load_single_image(image_path, args.image_size).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image)
            pred_box = outputs["localization"][0].cpu()
            # MultiTask output bbox is in pixel-space; normalize for plotting helper.
            h, w = image.shape[-2], image.shape[-1]
            pred_box = pred_box / torch.tensor([w, h, w, h], dtype=pred_box.dtype)
            pred_mask = outputs["segmentation"].argmax(dim=1)[0].cpu()
            pred_label = int(outputs["classification"].argmax(dim=1).item())
            # Log both interpretable outputs: box localization and dense mask prediction.
            log_payload[f"{image_path.stem}_bbox"] = make_wandb_image_from_bbox(image[0].cpu(), pred_box=pred_box, caption=f"{image_path.name} | breed={pred_label}")
            log_payload[f"{image_path.stem}_trimap"] = wandb.Image(mask_to_rgb(pred_mask), caption=f"{image_path.name} predicted_trimap")
            log_payload[f"{image_path.stem}_overlay"] = wandb.Image(overlay_to_uint8(image[0].cpu(), pred_mask), caption=f"{image_path.name} predicted_overlay")
        wandb.log(log_payload)
        run.finish()


if __name__ == "__main__":
    main()
