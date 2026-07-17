"""
Facial-expression affective stream (DSCT).

Wraps a Deformable-DETR-based DSCT model (Decoupled Subject/Context
Transformer) with a ResNet-50 backbone to localize and classify the
presenter's facial expression. The shared YOLO18-Pose detection is used
only to disambiguate which DSCT face proposal belongs to the tracked
presenter (via IoU matching), not as a model input.

This module depends on the external DSCT codebase (model definitions
and dataset transforms), which is not vendored into this repository.
Clone or place your local DSCT base repository under `third_party/DSCT/`
before using `DSCTFacialStream` (see `third_party/DSCT/README.md`).

Note on output: as implemented, this stream is an inference-time engine
around a pretrained DSCT checkpoint. `predict()` returns per-class
softmax probabilities over the checkpoint's trained label set (7 CAER
categories: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise, when
`dataset="caer"`) together with the matched bounding box - not an
intermediate embedding vector. If a different label space (e.g. the
26-category BoLD taxonomy) is required, retrain or fine-tune the
underlying DSCT checkpoint accordingly; this wrapper does not alter the
checkpoint's output head.
"""

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..preprocessing.yolo18_pose import YOLO18Pose  # noqa: F401  (re-exported for convenience)
from ...utils.box_ops import compute_box_iou, rescale_bboxes


class DSCTFacialStream:
    """
    Facial-expression inference engine built on a DSCT checkpoint.

    This class is an inference-time wrapper (not an `nn.Module`): it
    loads a pretrained DSCT model and a pretrained YOLO18-Pose model,
    and combines their outputs at inference time via IoU matching. It
    is not intended to be trained end-to-end through this wrapper.

    Args:
        dsct_repo_path: path to the external DSCT codebase (the
            directory containing `models/` and `datasets/`). Typically
            `third_party/DSCT` after placing your local clone there.
        dataset: dataset/label-space identifier used by the DSCT
            checkpoint (e.g. "caer").
        ckpt_path: path to the DSCT model checkpoint (.pth).
        yolo_path: path to a saved `YOLO18Pose` module (.pt), used only
            for bounding-box disambiguation.
        device: torch device string, e.g. "cuda" or "cpu".
    """

    def __init__(
        self,
        dsct_repo_path: str,
        dataset: str = 'caer',
        ckpt_path: str = '',
        yolo_path: str = '',
        device: str = 'cuda',
    ):
        sys.path.insert(0, str(Path(dsct_repo_path)))
        from models import build_model  # from the external DSCT repo
        from datasets.caer import make_face_transforms  # from the external DSCT repo

        self.device = torch.device(device)
        self.dataset = dataset
        self.emo_list = (
            ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            if dataset == 'caer' else []
        )
        self.transform = make_face_transforms("val")

        args = types.SimpleNamespace(
            dataset_file=dataset,
            binary_flag=0,
            num_queries=9,
            backbone='resnet50',
            detr='deformable_detr_dsct',
            model='deformable_transformer_dsct',
            dec_n_sp=0,
            dec_n_sm=0,
            pretrained_weights=ckpt_path,
            device=device,
            hidden_dim=256,
            nheads=8,
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            num_feature_levels=4,
            enc_n_points=4,
            dec_n_points=4,
            position_embedding='sine',
            masks=False,
        )
        self.model, _, _ = build_model(args)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.model.load_state_dict(
            {
                k.replace('class_embed.', 'class_embed_dsct.')
                 .replace('bbox_embed.', 'bbox_embed_dsct.'): v
                for k, v in sd.items()
            },
            strict=False,
        )
        self.model.to(self.device).eval()
        self.yolo18 = torch.load(yolo_path, map_location=self.device).eval()

    @torch.no_grad()
    def predict(self, frame: np.ndarray):
        """
        Run facial-expression inference on a single BGR video frame.

        Args:
            frame: a single frame as a BGR `numpy.ndarray` (as returned
                by `cv2.VideoCapture`).

        Returns:
            Tuple of:
                - probas: per-class softmax probabilities (shape
                  `(len(self.emo_list),)`) for the DSCT proposal that
                  best matches the YOLO18-Pose presenter box.
                - box: the matched proposal's bounding box in
                  `(x1, y1, x2, y2)` pixel coordinates.
        """
        y_out = self.yolo18(frame, verbose=False)
        y_box = (
            y_out[0].boxes.xyxy[0].cpu().numpy()
            if (len(y_out) > 0 and len(y_out[0].boxes) > 0) else None
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        w, h = pil_img.size
        img_t, _ = self.transform(
            pil_img, {"size": torch.as_tensor([h, w]), "orig_size": torch.as_tensor([h, w])}
        )

        out = self.model(img_t.unsqueeze(0).to(self.device))
        probas = out['pred_logits'][0, :, :-1].softmax(-1).cpu().numpy()
        boxes = rescale_bboxes(out['pred_boxes'][0].cpu(), pil_img.size).numpy()

        best_idx, max_s = 0, -1.0
        for i, box in enumerate(boxes):
            conf = np.max(probas[i])
            score = (compute_box_iou(y_box, box) + conf) if y_box is not None else conf
            if score > max_s:
                max_s, best_idx = score, i
        return probas[best_idx], boxes[best_idx]
