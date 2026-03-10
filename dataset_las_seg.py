# dataset_las_seg.py
"""
LAS/LAZ 数据集（语义分割）——针对 LAS 特征定制版

核心能力：
1) 读取 xyz（必要）
2) 读取可选输入特征：intensity、return_number、number_of_returns、rgb、classification 等
3) 标签字段自动选择：
   - 优先 ExtraBytes：label / sem_label / plane_id / ...
   - 若没有 ExtraBytes 标签，则用 classification 作为标签（常见）
4) 自动把原始标签 remap 到 [0, num_classes-1] 连续整数（训练必须）
5) 固定点数采样、归一化、增强
6) 返回：
   xyz:      (num_points, 3)
   feats:    (num_points, Din)  若Din=0则返回 None
   labels:   (num_points,)
"""

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import laspy

# ----------------------------
# 一些小工具函数
# ----------------------------

def _random_rotate_z(xyz: np.ndarray) -> np.ndarray:
    """绕 Z 轴随机旋转（很多地形/岩体点云适用）"""
    theta = np.random.uniform(0, 2*np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)
    return xyz @ R.T

def _normalize_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """
    归一化到单位球：
    - 减去质心（center）
    - 除以最大半径（scale）
    """
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = np.max(np.linalg.norm(xyz, axis=1))
    xyz = xyz / (scale + 1e-6)
    return xyz.astype(np.float32)

def _dims_set(las: laspy.LasData) -> set:
    """LAS 当前文件拥有的所有维度名集合（包含 extra bytes）"""
    return set(las.point_format.dimension_names)

def _find_label_field(las: laspy.LasData) -> str | None:
    """
    自动寻找“标签字段”：
    - 先看 extra bytes 里有没有常见命名的 label 字段
    - 没有则返回 None（表示后续用 classification 作为标签）
    """
    candidates = ["label", "labels", "sem_label", "semantic", "plane_id", "seg", "gt"]
    dims = _dims_set(las)
    for name in candidates:
        if name in dims:
            return name
    return None

def _read_xyz(las: laspy.LasData) -> np.ndarray:
    """
    las.x/las.y/las.z 是应用过 scale/offset 的真实坐标（float）
    shape: (N,3)
    """
    return np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)

def _read_rgb_if_exists(las: laspy.LasData) -> np.ndarray | None:
    """
    读取 RGB（如果存在 red/green/blue 维度）。
    注意：
    - LAS 的 RGB 常是 uint16（0~65535），你可归一化到 0~1
    """
    dims = _dims_set(las)
    if {"red", "green", "blue"}.issubset(dims):
        rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32)
        # 归一化到 0~1（避免值过大影响训练）
        rgb = rgb / 65535.0
        return rgb
    return None

def _read_feature(las: laspy.LasData, name: str) -> np.ndarray:
    """
    从 LAS 读取一个标量特征（返回 shape=(N,1)）。
    对一些字段做合理归一化/缩放以稳定训练。
    """
    dims = _dims_set(las)

    if name == "intensity":
        if "intensity" not in dims:
            raise KeyError("LAS has no 'intensity'")
        x = np.asarray(las.intensity).astype(np.float32)
        # intensity 常见范围 0~65535，归一化到 0~1
        x = x / 65535.0
        return x[:, None]

    if name == "return_number":
        # return_number 是 1..num_returns，归一化到 0~1
        if "return_number" not in dims:
            raise KeyError("LAS has no 'return_number'")
        x = np.asarray(las.return_number).astype(np.float32)
        # 简单除以 7 或 8（多数激光雷达回波数很小），也可按数据统计做更合理归一化
        x = x / 8.0
        return x[:, None]

    if name == "num_returns":
        # number_of_returns
        if "number_of_returns" not in dims:
            raise KeyError("LAS has no 'number_of_returns'")
        x = np.asarray(las.number_of_returns).astype(np.float32)
        x = x / 8.0
        return x[:, None]

    if name == "classification":
        if "classification" not in dims:
            raise KeyError("LAS has no 'classification'")
        # classification 作为输入特征时，把它缩放到 0~1
        x = np.asarray(las.classification).astype(np.float32) / 255.0
        return x[:, None]

    raise ValueError(f"Unknown feature name: {name}")

def _read_labels(las: laspy.LasData) -> np.ndarray:
    """
    读取“原始标签”（还未 remap）
    - 优先 extra bytes label
    - 否则 classification
    """
    label_field = _find_label_field(las)
    if label_field is not None:
        return np.asarray(getattr(las, label_field)).astype(np.int64)
    return np.asarray(las.classification).astype(np.int64)

# ----------------------------
# 标签 remap：把原始 label 映射到 0..C-1
# ----------------------------

def build_label_mapping(all_label_values: np.ndarray, save_path: str | None = None) -> dict:
    """
    给定所有样本出现过的原始标签集合，构建连续映射表：
        raw_label -> new_label_id
    e.g., {2:0, 5:1, 9:2}
    """
    uniq = np.unique(all_label_values)
    uniq = uniq.tolist()
    mapping = {int(v): i for i, v in enumerate(uniq)}

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"raw_to_new": mapping, "num_classes": len(mapping)}, f, ensure_ascii=False, indent=2)
    return mapping

def remap_labels(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """把 labels 逐个映射到新 id。"""
    out = np.empty_like(labels, dtype=np.int64)
    for k, v in mapping.items():
        out[labels == k] = int(v)
    return out

# ----------------------------
# Dataset
# ----------------------------

class LASPointSegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        num_points: int = 4096,
        feature_names: list[str] | None = None,
        augment: bool = True,
        allow_laz: bool = True,
        label_mapping: dict | None = None,
        cache_label_mapping_path: str | None = None,
    ):
        """
        root_dir: 数据根目录 data/rockplane_las
        split: train/val/test
        num_points: 每个样本采样点数（固定）
        feature_names: 你想用哪些 LAS 字段作为输入特征（不含 xyz）
                       例如 ["intensity", "return_number", "num_returns", "rgb"]
        augment: 是否做增强（仅 train 有效）
        label_mapping: 传入 raw->new 映射；若为 None 且 split==train，会自动扫描 train 构建映射
        cache_label_mapping_path: 映射缓存 json 的保存路径（方便复现实验）
        """
        assert split in ["train", "val", "test"]
        self.split = split
        self.num_points = num_points
        self.feature_names = feature_names or []
        self.augment = augment and (split == "train")

        # 收集文件
        exts = ["*.las"]
        if allow_laz:
            exts.append("*.laz")

        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(root_dir, split, e)))
        self.files = sorted(files)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No LAS/LAZ found in: {os.path.join(root_dir, split)}")

        # label mapping：训练集可以自动构建；val/test 必须复用 train 的 mapping
        self.label_mapping = label_mapping
        if self.label_mapping is None and split == "train":
            # 扫描 train 集所有 raw label，构建映射（确保训练 label 连续）
            all_labels = []
            for p in self.files:
                las = laspy.read(p)
                all_labels.append(_read_labels(las))
            all_labels = np.concatenate(all_labels, axis=0)
            self.label_mapping = build_label_mapping(all_labels, save_path=cache_label_mapping_path)

    def __len__(self):
        return len(self.files)

    def _build_features(self, las: laspy.LasData) -> np.ndarray | None:
        """
        根据 feature_names 拼接特征矩阵 feats: (N, Din)
        - "rgb" 特殊：一次性提供 3 维
        - 其余特征一般是标量 1 维
        """
        if len(self.feature_names) == 0:
            return None

        feats_list = []
        for name in self.feature_names:
            if name == "rgb":
                rgb = _read_rgb_if_exists(las)
                if rgb is None:
                    raise KeyError("You requested feature 'rgb' but LAS has no red/green/blue.")
                feats_list.append(rgb)  # (N,3)
            else:
                feats_list.append(_read_feature(las, name))  # (N,1)

        feats = np.concatenate(feats_list, axis=1).astype(np.float32)  # (N,Din)
        return feats

    def __getitem__(self, idx):
        path = self.files[idx]
        fname = os.path.basename(path)

        las = laspy.read(path)

        # 1) 读取 xyz
        xyz = _read_xyz(las)  # (N,3)

        # 2) 读取 features（可选）
        feats = self._build_features(las)  # (N,Din) or None

        # 3) 读取 labels（原始）
        raw_labels = _read_labels(las).astype(np.int64)  # (N,)

        # 4) 固定点数采样（同时对 xyz/feats/labels 同步采样）
        N = xyz.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.random.choice(N, self.num_points, replace=True)

        xyz = xyz[choice]
        raw_labels = raw_labels[choice]
        if feats is not None:
            feats = feats[choice]

        # 5) 归一化 xyz（非常重要）
        xyz = _normalize_unit_sphere(xyz)

        # 6) 数据增强（仅训练）
        if self.augment:
            xyz = _random_rotate_z(xyz)
            xyz += np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)

        # 7) remap label 到连续 id（val/test 必须有 mapping）
        if self.label_mapping is None:
            raise RuntimeError("label_mapping is None. For val/test you must pass train mapping.")
        labels = remap_labels(raw_labels, self.label_mapping)

        # 8) 转为 torch
        xyz_t = torch.from_numpy(xyz).float()               # (P,3)
        feats_t = torch.from_numpy(feats).float() if feats is not None else None
        labels_t = torch.from_numpy(labels).long()          # (P,)

        return xyz_t, feats_t, labels_t, fname