"""
Skeleton-based affective stream (InfoGCN).

This module implements the skeleton-based encoder used to process
body-keypoint sequences. It is built on an InfoGCN-style graph
convolutional backbone: each spatial block combines a fixed, learnable
graph adjacency with a data-driven spatial-attention term, followed by
a temporal convolution over the frame axis. A gain/shift bottleneck is
applied to the pooled representation before the final classification
head.

Expected input shape: (N, C, T, V)
    N - batch size
    C - channels per joint (e.g. x, y[, confidence])
    T - number of frames in the sequence
    V - number of joints (skeleton graph nodes)

Note on the joint graph: `InfoGCNSkeletonStream` requires the caller to
supply `pairs`, the list of (joint_i, joint_j) edges defining skeletal
connectivity for the chosen keypoint topology (e.g. the 18-keypoint
layout produced by `grace.models.preprocessing.yolo18_pose.YOLO18Pose`,
which adds an interpolated neck/SSN joint to the standard 17-keypoint
COCO layout). No default topology is assumed here; see
`configs/skeleton_config.yaml` for an example edge list.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adjacency(num_joints: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build a symmetrically-normalized adjacency matrix from a list of
    skeletal joint connections.

    Args:
        num_joints: total number of joints/nodes in the skeleton graph.
        pairs: list of (i, j) index pairs representing bone connections.

    Returns:
        A (num_joints, num_joints) symmetrically-normalized adjacency
        matrix, including self-loops.
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in pairs:
        A[i, j] = 1
        A[j, i] = 1
    A += np.eye(num_joints, dtype=np.float32)
    D = np.diag(A.sum(axis=1) ** -0.5)
    return D @ A @ D


class InfoTemporalConv(nn.Module):
    """Temporal convolution applied independently at each joint."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (kernel, 1), (stride, 1), (self.pad, 0))
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class InfoGCN_SpatialBlock(nn.Module):
    """
    Spatial graph-convolution block combining a fixed learnable
    adjacency with a data-driven spatial-attention affinity, following
    an InfoGCN-style multi-subset formulation.
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor, num_subsets: int = 3):
        super().__init__()
        self.num_subsets = num_subsets
        self.num_joints = A.shape[1]
        self.PA = nn.Parameter(A.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        attn_ch = max(out_ch // 4, 1)
        self.conv_query = nn.Conv2d(in_ch, attn_ch, 1)
        self.conv_key = nn.Conv2d(in_ch, attn_ch, 1)
        self.attn_scale = attn_ch ** -0.5
        self.conv_d = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 1) for _ in range(num_subsets)])
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        q = self.conv_query(x).mean(dim=2)
        k = self.conv_key(x).mean(dim=2)
        A_attn = torch.einsum('ncu,ncv->nuv', q, k) * self.attn_scale
        A_attn = torch.softmax(A_attn, dim=-1)
        y = None
        for i in range(self.num_subsets):
            A_base = self.PA[i] + self.alpha * self.PA[i]
            z = torch.einsum('nctv,vw->nctw', x, A_base)
            z = z + torch.einsum('nctv,nvw->nctw', x, A_attn)
            z = self.conv_d[i](z)
            y = z + y if y is not None else z
        y = self.bn(y) + self.down(x)
        return self.relu(y)


class InfoGCNBlock(nn.Module):
    """Spatial block followed by temporal convolution, with a residual path."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor, stride: int = 1):
        super().__init__()
        self.gcn = InfoGCN_SpatialBlock(in_ch, out_ch, A)
        self.tcn = InfoTemporalConv(out_ch, out_ch, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.MaxPool2d((stride, 1)) if stride != 1 else nn.Identity(),
            )
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gcn_feat = self.gcn(x)
        tcn_feat = self.tcn(gcn_feat)
        down_feat = self.down(x)
        if tcn_feat.shape[2] != down_feat.shape[2]:
            down_feat = F.interpolate(
                down_feat, size=(tcn_feat.shape[2], tcn_feat.shape[3]), mode='nearest'
            )
        return self.relu(tcn_feat + down_feat)


class InfoGCNSkeletonStream(nn.Module):
    """
    Skeleton-based affective encoder built on an InfoGCN-style graph
    convolutional backbone.

    Consumes normalized joint-coordinate sequences of shape (N, C, T, V)
    and outputs class logits of shape (N, num_classes) (e.g. 26-way
    multi-label affect logits when trained on BoLD; pair with
    `nn.BCEWithLogitsLoss` for multi-label supervision).

    A gain/shift bottleneck (`bottleneck_gain`, `bottleneck_shift`) is
    applied to the pooled feature vector before the final linear
    classifier, modulating the representation via a learned
    element-wise affine transform.
    """

    def __init__(
        self,
        num_joints: int,
        num_classes: int,
        pairs: List[Tuple[int, int]],
        in_channels: int,
        base_hidden: int = 64,
    ):
        super().__init__()
        A_np = build_adjacency(num_joints, pairs)
        A3 = np.stack([A_np, A_np, A_np], axis=0).astype(np.float32)
        self._A = torch.tensor(A3)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        b = base_hidden
        self.layers = nn.ModuleList([
            InfoGCNBlock(in_channels, b, self._A),
            InfoGCNBlock(b, b, self._A),
            InfoGCNBlock(b, b, self._A),
            InfoGCNBlock(b, b, self._A),
            InfoGCNBlock(b, b * 2, self._A, stride=2),
            InfoGCNBlock(b * 2, b * 2, self._A),
            InfoGCNBlock(b * 2, b * 2, self._A),
            InfoGCNBlock(b * 2, b * 4, self._A, stride=2),
            InfoGCNBlock(b * 4, b * 4, self._A),
            InfoGCNBlock(b * 4, b * 4, self._A),
        ])
        self.bottleneck_gain = nn.Linear(b * 4, b * 4)
        self.bottleneck_shift = nn.Linear(b * 4, b * 4)
        nn.init.constant_(self.bottleneck_gain.weight, 0.0)
        nn.init.constant_(self.bottleneck_gain.bias, 1.0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.6)
        self.fc = nn.Linear(b * 4, num_classes)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self._A = self._A.to(*args, **kwargs)
        for layer in self.layers:
            layer.gcn.PA.data = layer.gcn.PA.data.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).view(N, -1)
        gain = self.bottleneck_gain(x)
        shift = self.bottleneck_shift(x)
        x = x * gain + shift
        return self.fc(self.drop(x))
