"""
YOLO18-Pose: shared presenter-localization and keypoint pre-processing.

Wraps a standard Ultralytics YOLO-Pose model and augments its 17-keypoint
output with an 18th interpolated joint (Suprasternal Notch / neck),
computed as a confidence-weighted blend of the nose, ear, and shoulder
keypoints. This shared pre-processing stage feeds both the skeleton
stream (`grace.models.skeleton.infogcn`) and, via bounding-box matching,
the facial stream (`grace.models.facial.dsct`).
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints


class YOLO18Pose(nn.Module):
    """
    YOLO-Pose wrapper that adds an interpolated 18th (neck/SSN) keypoint
    to the standard 17-keypoint COCO output.

    The neck point is computed as a confidence-weighted blend of the
    left shoulder, right shoulder, nose, and both ears, using the fixed
    weights in `WEIGHTS`. Its confidence is set to the minimum of the
    two shoulder confidences, since the estimate is most reliable when
    both shoulders are confidently detected.
    """

    NOSE, LEFT_EAR, RIGHT_EAR, L_SHOULDER, R_SHOULDER = 0, 3, 4, 5, 6
    WEIGHTS = {'l_shoulder': 0.35, 'r_shoulder': 0.35, 'nose': 0.20, 'l_ear': 0.05, 'r_ear': 0.05}

    def __init__(self, weights_path: str):
        """
        Args:
            weights_path: path to a YOLO-Pose checkpoint (.pt) compatible
                with the standard 17-keypoint COCO layout.
        """
        super().__init__()
        self.yolo = YOLO(weights_path)

    def _estimate_neck(self, xy: torch.Tensor, conf: torch.Tensor):
        """
        Interpolate the 18th (neck/SSN) keypoint from surrounding joints.

        Args:
            xy: (N, 17, 2) keypoint coordinates.
            conf: (N, 17) per-keypoint confidence scores.

        Returns:
            Tuple of (neck_xy, neck_conf) with shapes (N, 1, 2) and (N, 1).
        """
        ls_xy, rs_xy = xy[:, self.L_SHOULDER], xy[:, self.R_SHOULDER]
        nos_xy = xy[:, self.NOSE]
        ls_c, rs_c, nos_c = conf[:, self.L_SHOULDER], conf[:, self.R_SHOULDER], conf[:, self.NOSE]
        le_c, re_c = conf[:, self.LEFT_EAR], conf[:, self.RIGHT_EAR]

        w_ls = self.WEIGHTS['l_shoulder'] * ls_c
        w_rs = self.WEIGHTS['r_shoulder'] * rs_c
        w_nos = self.WEIGHTS['nose'] * nos_c
        total_w = torch.clamp(
            w_ls + w_rs + w_nos + (self.WEIGHTS['l_ear'] * le_c) + (self.WEIGHTS['r_ear'] * re_c),
            min=1e-6,
        )

        neck_x = (w_ls * ls_xy[:, 0] + w_rs * rs_xy[:, 0] + w_nos * nos_xy[:, 0]) / total_w
        neck_y = (w_ls * ls_xy[:, 1] + w_rs * rs_xy[:, 1] + w_nos * nos_xy[:, 1]) / total_w

        neck_xy = torch.stack([neck_x, neck_y], dim=1).unsqueeze(1)
        neck_conf = torch.min(ls_c, rs_c).unsqueeze(1)
        return neck_xy, neck_conf

    def forward(self, source, **kwargs):
        """
        Run YOLO-Pose inference and append the interpolated 18th keypoint
        to each detected person's keypoint set.

        Args:
            source: image, path, or batch input accepted by Ultralytics YOLO.
            **kwargs: forwarded to the underlying `YOLO.__call__`.

        Returns:
            The list of Ultralytics `Results` objects, with `.keypoints`
            extended from 17 to 18 points per detection.
        """
        results = self.yolo(source, **kwargs)
        for r in results:
            if r.keypoints is not None and len(r.keypoints) > 0:
                neck_xy, neck_conf = self._estimate_neck(r.keypoints.xy, r.keypoints.conf)
                r.keypoints = Keypoints(
                    torch.cat(
                        [
                            torch.cat([r.keypoints.xy, neck_xy], dim=1),
                            torch.cat([r.keypoints.conf, neck_conf], dim=1).unsqueeze(-1),
                        ],
                        dim=-1,
                    ),
                    orig_shape=r.orig_shape,
                )
        return results
