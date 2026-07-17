"""
Evaluation entry point for either affective stream.

Usage:
    python scripts/evaluate.py --stream skeleton --config configs/skeleton_config.yaml --checkpoint checkpoints/skeleton/epoch_100.pt
    python scripts/evaluate.py --stream facial   --config configs/facial_config.yaml

Note: the skeleton stream is evaluated in batch mode (its forward pass
accepts a batch tensor directly). The facial stream, as implemented in
`DSCTFacialStream`, is an inference engine that processes one raw video
frame at a time via `.predict()`, so it is evaluated with a simple
per-sample loop instead.
"""

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from grace.data.dataset import BoldStreamDataset
from grace.models.facial.dsct import DSCTFacialStream
from grace.models.skeleton.infogcn import InfoGCNSkeletonStream
from grace.utils.logger import get_logger
from grace.utils.metrics import macro_f1, mean_auc_roc, mean_average_precision, per_class_ap

logger = get_logger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a GRACE affective stream.")
    parser.add_argument("--stream", type=str, choices=["skeleton", "facial"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Required for --stream skeleton; ignored for facial "
                              "(the facial config already points at a checkpoint).")
    parser.add_argument("--class-names", type=str, nargs="*", default=None,
                         help="Optional list of class names for per-class AP reporting.")
    return parser.parse_args()


def evaluate_skeleton(cfg, checkpoint_path):
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    model = InfoGCNSkeletonStream(
        num_joints=cfg["model"]["num_joints"],
        num_classes=cfg["model"]["num_classes"],
        pairs=[tuple(p) for p in cfg["model"]["pairs"]],
        in_channels=cfg["model"]["in_channels"],
        base_hidden=cfg["model"]["base_hidden"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    val_set = BoldStreamDataset(
        root=cfg["data"]["root"], split="val", mode="skeleton", labels_csv=cfg["data"]["labels_csv"]
    )
    val_loader = DataLoader(val_set, batch_size=cfg["data"]["batch_size"], shuffle=False)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(y.numpy())

    y_score = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = (y_score >= 0.5).astype(np.float32)

    logger.info(f"mAP:      {mean_average_precision(y_true, y_score):.4f}")
    logger.info(f"macro-F1: {macro_f1(y_true, y_pred):.4f}")
    logger.info(f"mean AUC: {mean_auc_roc(y_true, y_score):.4f}")


def evaluate_facial(cfg):
    stream = DSCTFacialStream(
        dsct_repo_path=cfg["paths"]["dsct_repo_path"],
        dataset=cfg["model"]["dataset"],
        ckpt_path=cfg["paths"]["ckpt_path"],
        yolo_path=cfg["paths"]["yolo18_pose_path"],
        device=cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu",
    )

    val_set = BoldStreamDataset(
        root=cfg.get("data", {}).get("root", ""),
        split="val",
        mode="facial",
        labels_csv=cfg.get("data", {}).get("labels_csv", ""),
    )

    all_scores, all_labels = [], []
    for frame, y in val_set:
        probas, _box = stream.predict(frame)
        all_scores.append(probas)
        all_labels.append(y.numpy())

    y_score = np.stack(all_scores, axis=0)
    y_true = np.stack(all_labels, axis=0)
    y_pred = (y_score >= 0.5).astype(np.float32)

    logger.info(f"mAP:      {mean_average_precision(y_true, y_score):.4f}")
    logger.info(f"macro-F1: {macro_f1(y_true, y_pred):.4f}")
    logger.info(f"mean AUC: {mean_auc_roc(y_true, y_score):.4f}")


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.stream == "skeleton":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --stream skeleton")
        evaluate_skeleton(cfg, args.checkpoint)
    else:
        evaluate_facial(cfg)


if __name__ == "__main__":
    main()
