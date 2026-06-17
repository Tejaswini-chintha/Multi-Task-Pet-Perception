"""Question 2.6: log segmentation examples and compare Dice vs pixel accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.question2.common import build_args, load_model, make_dataloaders, overlay_mask, wandb
from train import dice_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="oxford-iiit-pet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0, help="Limit number of validation batches (0 = all)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--project", default="da6401-assignment-2")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def mask_to_rgb(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    color_map = np.array(
        [
            [255, 0, 0],    # class 0
            [0, 255, 0],    # class 1
            [0, 0, 255],    # class 2
        ],
        dtype=np.float32,
    ) / 255.0
    return (color_map[np.clip(mask, 0, 2)] * 255.0).astype(np.uint8)


def image_tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    # Reverse ImageNet normalization used by the dataset transform.
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
    image = image_tensor.detach().cpu() * std + mean
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def overlay_to_uint8(image_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> np.ndarray:
    overlay = np.clip(overlay_mask(image_tensor, mask_tensor), 0.0, 1.0)
    return (overlay * 255.0).astype(np.uint8)


def evaluate_segmentation(model, loader, device, max_batches: int = 0):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses, dice_values, pixel_acc_values = [], [], []
    # Keep a fixed-size qualitative panel for report visuals.
    collected_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            images = batch["image"].to(device)
            targets = batch["segmentation_mask"].to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            losses.append(float(criterion(logits, targets).item()))
            dice_values.append(float(dice_score(logits.detach().cpu(), batch["segmentation_mask"]).item()))
            pixel_acc_values.append(pixel_accuracy(logits.detach().cpu(), batch["segmentation_mask"]))
            preds_cpu = preds.detach().cpu()
            for idx in range(preds_cpu.size(0)):
                if len(collected_samples) >= 5:
                    break
                collected_samples.append(
                    {
                        "image_id": batch["image_id"][idx],
                        "image": batch["image"][idx].detach().cpu(),
                        "target": batch["segmentation_mask"][idx].detach().cpu(),
                        "pred": preds_cpu[idx],
                    }
                )

    metrics = {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "dice": float(sum(dice_values) / max(len(dice_values), 1)),
        "pixel_accuracy": float(sum(pixel_acc_values) / max(len(pixel_acc_values), 1)),
    }
    return metrics, collected_samples


def main():
    args = parse_args()
    model = load_model("segmentation", args.checkpoint)
    device = next(model.parameters()).device
    train_loader, val_loader = make_dataloaders(
        build_args(data_root=args.data_root, image_size=args.image_size, batch_size=args.batch_size, task="segmentation")
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run = None
    if wandb is not None and not args.disable_wandb:
        run = wandb.init(project=args.project, name="q2_6_dice_vs_pixel_accuracy")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_batches > 0 and batch_idx > args.max_batches:
                break
            images = batch["image"].to(device)
            targets = batch["segmentation_mask"].to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Recompute validation metrics each epoch so W&B shows proper val curves.
        val_metrics, collected_samples = evaluate_segmentation(model, val_loader, device, max_batches=args.max_batches)
        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_dice={val_metrics['dice']:.4f} | "
            f"val_pixel_acc={val_metrics['pixel_accuracy']:.4f}"
        )
        if run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/dice_score": val_metrics["dice"],
                    "val/pixel_accuracy": val_metrics["pixel_accuracy"],
                }
            )

    if run is not None:
        sample_payload = []
        for sample in collected_samples:
            image_id = sample["image_id"]
            sample_payload.append(
                [
                    image_id,
                    wandb.Image(image_tensor_to_uint8(sample["image"]), caption=f"{image_id} original"),
                    wandb.Image(mask_to_rgb(sample["target"]), caption=f"{image_id} gt_trimap"),
                    wandb.Image(mask_to_rgb(sample["pred"]), caption=f"{image_id} pred_trimap"),
                    wandb.Image(overlay_to_uint8(sample["image"], sample["pred"]), caption=f"{image_id} pred_overlay"),
                ]
            )
        table = wandb.Table(
            columns=["image_id", "original_image", "ground_truth_trimap", "predicted_trimap", "predicted_overlay"],
            data=sample_payload,
        )
        wandb.log({"segmentation_examples": table})
        run.finish()


if __name__ == "__main__":
    main()
