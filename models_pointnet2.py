# models_pointnet2.py
"""
PointNet++ (SSG) 语义分割模型：
- 输入：
    xyz:      (B, N, 3)
    features: (B, N, Din) 或 None
  注意：features 不包含 xyz；xyz 永远单独传入用于几何分组。

- 输出：
    logits: (B, N, num_classes)

你可以通过 Din 来适配 LAS 的额外属性：
比如 Din=1 (intensity) 或 Din=1+1+1+3 等。
"""

import torch
import torch.nn as nn
from pointnet2_utils import sample_and_group, three_nn, three_interpolate

def conv_bn_relu(in_c, out_c):
    """2D 1x1 conv + BN + ReLU，用于处理 grouped points 的 (B,C,nsample,S)"""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

def conv1d_bn_relu(in_c, out_c):
    """1D 1x1 conv + BN + ReLU，用于处理 (B,C,N)"""
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, 1, bias=False),
        nn.BatchNorm1d(out_c),
        nn.ReLU(inplace=True),
    )

class PointNetSetAbstraction(nn.Module):
    """
    SA 层：采样 npoint 个中心点 + ball query 分组 + PointNet(MLP+maxpool) 提取局部特征
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_c = in_channel
        for out_c in mlp_channels:
            layers.append(conv_bn_relu(last_c, out_c))
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3)
        points: (B, N, D) or None
        return:
          new_xyz: (B, S, 3)
          new_points: (B, S, C)
        """
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_points: (B,S,nsample,3+D) -> (B,3+D,nsample,S)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)              # (B,C,nsample,S)
        new_points = torch.max(new_points, dim=2)[0]   # (B,C,S)
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B,S,C)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    """
    FP 层：从稀疏层插值回密集层，并与 skip feature 拼接，再用 MLP 融合。
    """
    def __init__(self, in_channel, mlp_channels):
        super().__init__()
        layers = []
        last_c = in_channel
        for out_c in mlp_channels:
            layers.append(conv1d_bn_relu(last_c, out_c))
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: (B, N, 3)  目标（更密）
        xyz2: (B, S, 3)  来源（更稀）
        points1: (B, N, C1) skip feature，可为 None
        points2: (B, S, C2) source feature
        return: (B, N, C')
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dist, idx = three_nn(xyz1, xyz2)
            interpolated_points = three_interpolate(points2, idx, dist)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1).contiguous()  # (B,C,N)
        new_points = self.mlp(new_points)
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B,N,C')
        return new_points

class PointNet2SemSeg(nn.Module):
    """
    PointNet++ SSG 语义分割。
    关键：Din 是“额外输入特征”的维度（不含 xyz）。
    SA1 的 in_channel = 3 + Din（因为 sample_and_group 会拼接相对 xyz(3) + features(Din)）
    """
    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()

        # SA 层参数（岩体点云可按实际尺度调整 radius）
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.10, nsample=32, in_channel=3 + feature_dim, mlp_channels=[32, 32, 64])
        self.sa2 = PointNetSetAbstraction(npoint=256,  radius=0.20, nsample=32, in_channel=3 + 64,         mlp_channels=[64, 64, 128])
        self.sa3 = PointNetSetAbstraction(npoint=64,   radius=0.40, nsample=32, in_channel=3 + 128,        mlp_channels=[128, 128, 256])
        self.sa4 = PointNetSetAbstraction(npoint=16,   radius=0.80, nsample=32, in_channel=3 + 256,        mlp_channels=[256, 256, 512])

        # FP（反向传播到原始点）
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp_channels=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp_channels=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64,  mlp_channels=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128,       mlp_channels=[128, 128, 128])

        # 每点分类头
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz, features=None):
        """
        xyz: (B, N, 3)
        features: (B, N, Din) 或 None
        """
        l0_xyz = xyz
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)                 # (B,1024,3), (B,1024,64)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)                 # (B,256,3),  (B,256,128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)                 # (B,64,3),   (B,64,256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)                 # (B,16,3),   (B,16,512)

        l3_fp = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)          # (B,64,256)
        l2_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_fp)              # (B,256,256)
        l1_fp = self.fp2(l1_xyz, l2_xyz, l1_points, l2_fp)              # (B,1024,128)
        l0_fp = self.fp1(l0_xyz, l1_xyz, None, l1_fp)                   # (B,N,128)

        x = l0_fp.permute(0, 2, 1).contiguous()                         # (B,128,N)
        logits = self.classifier(x).permute(0, 2, 1).contiguous()        # (B,N,C)
        return logits