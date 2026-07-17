"""
Training entry point for the InfoGCN skeleton stream.

The provided InfoGCN source file did not include a training loop, so
this script is a standard, generic PyTorch training loop written
around `InfoGCNSkeletonStream`. It is not a verified reproduction of
any specific paper's training run - review and adjust the loss,
optimizer, and schedule to match your actual experimental setup before
relying on its results.

Usage:
    python scripts/train_skeleton.py --config configs/skeleton_config.yaml
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from grace.data.dataset import BoldStreamDataset
from grace.models.skeleton.infogcn import InfoGCNSkeletonStream
from grace.utils.logger import get_logger

logger = get_logger("train_skeleton")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the InfoGCN skeleton stream.")
    parser.add_argument(
        "--config", type=str, default="configs/skeleton_config.yaml",
        help="Path to the skeleton stream YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    model = InfoGCNSkeletonStream(
        num_joints=cfg["model"]["num_joints"],
        num_classes=cfg["model"]["num_classes"],
        pairs=[tuple(p) for p in cfg["model"]["pairs"]],
        in_channels=cfg["model"]["in_channels"],
        base_hidden=cfg["model"]["base_hidden"],
    ).to(device)

    train_set = BoldStreamDataset(
        root=cfg["data"]["root"],
        split="train",
        mode="skeleton",
        labels_csv=cfg["data"]["labels_csv"],
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"]["lr_decay_every"],
        gamma=cfg["train"]["lr_decay_factor"],
    )
    # Multi-label affect classification (26 BoLD categories): binary
    # cross-entropy over independent sigmoid outputs.
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        logger.info(f"epoch {epoch + 1}/{cfg['train']['epochs']} | loss: {running_loss / len(train_loader):.4f}")

        torch.save(model.state_dict(), f"{cfg['train']['checkpoint_dir']}/epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    main()
