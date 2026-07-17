"""
Bounding-box geometry utilities.

Used to match facial-detector proposals against the shared presenter
bounding box produced by YOLO18-Pose (see `grace.models.facial.dsct`).
"""

import torch


def compute_box_iou(box1, box2) -> float:
    """
    Intersection-over-union between two axis-aligned boxes in
    (x1, y1, x2, y2) format.
    """
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    b1A = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2A = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(b1A + b2A - interArea + 1e-6)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) boxes to (x1, y1, x2, y2) format."""
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([(x_c - .5 * w), (y_c - .5 * h), (x_c + .5 * w), (y_c + .5 * h)], dim=1)


def rescale_bboxes(out_bbox: torch.Tensor, size) -> torch.Tensor:
    """
    Rescale normalized (cx, cy, w, h) boxes to absolute pixel
    (x1, y1, x2, y2) coordinates for an image of the given size.

    Args:
        out_bbox: (N, 4) normalized boxes in (cx, cy, w, h) format.
        size: (img_w, img_h) tuple.
    """
    img_w, img_h = size
    return box_cxcywh_to_xyxy(out_bbox) * torch.tensor(
        [img_w, img_h, img_w, img_h], dtype=torch.float32
    )
