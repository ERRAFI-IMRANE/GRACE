"""
Dataset loader stub for BoLD-style video clips.

Neither of the source files this repository was built from included a
data-loading pipeline, so this module is a documented starting point
rather than a verified, ready-to-run loader. Adapt the paths and
extraction logic to your local copy of the Body Language Dataset (BoLD)
before use.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class BoldStreamDataset(Dataset):
    """
    Generic dataset for loading either skeleton keypoint sequences or
    facial crop sequences from a BoLD-style directory layout.

    Args:
        root: root directory containing the extracted clip data.
        split: one of "train", "val", "test".
        mode: "skeleton" to yield joint-coordinate tensors of shape
            (C, T, V); "facial" to yield facial crop tensors of shape
            (T, 3, H, W).
        labels_csv: path to the BoLD annotation CSV for this split
            (columns typically include the 26 discrete affect scores
            plus valence/arousal/dominance).
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"],
        mode: Literal["skeleton", "facial"],
        labels_csv: str,
    ):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.labels_csv = Path(labels_csv)

        # TODO: adapt to your local BoLD directory structure, e.g.:
        #   BoLD/annotations/{train,val,test}.csv
        #   BoLD/joints/003/{video_id}.mp4/{clip_id}.npy      (skeleton mode)
        #   BoLD/faces/{video_id}/{clip_id}/frame_%05d.jpg    (facial mode)
        self.samples = self._load_index()

    def _load_index(self):
        # TODO: parse `self.labels_csv` and build a list of
        # (clip_path, label_vector) tuples for this split.
        raise NotImplementedError(
            "BoldStreamDataset._load_index is a stub. Implement indexing "
            "against your local BoLD annotation files before use."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        clip_path, label = self.samples[idx]

        if self.mode == "skeleton":
            # Expected: an (C, T, V) array of normalized joint coordinates.
            arr = np.load(clip_path)
            x = torch.from_numpy(arr).float()
        else:
            # Expected: a (T, 3, H, W) array of aligned facial crops.
            arr = np.load(clip_path)
            x = torch.from_numpy(arr).float()

        y = torch.from_numpy(np.asarray(label, dtype=np.float32))
        return x, y
