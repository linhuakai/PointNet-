# -*- coding: utf-8 -*-
"""
Region Growing for Rock Structural Planes (LAS -> LAS) - Global (no label grouping)
带自动上色版本：
- 每个结构面簇自动分配唯一 RGB 颜色
- 输出主 LAS 带 segment_id / segment_class / segment_local / RGB
"""

import os
import sys
import csv
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Tuple, Optional, List, Dict
from sklearn.neighbors import NearestNeighbors
try:
    import laspy
except Exception as e:
    raise RuntimeError("需要安装 laspy：pip install laspy") from e


# ======================
# 参数
# ======================
@dataclass
class RGParams:
    k_neighbors: int = 50
    theta_deg: float = 40
    curv_thresh: float = 40
    seed_curv_percentile: float = 40
    min_cluster_size: int = 100
    use_seed_normal: bool = True
    curv_auto_percentile: Optional[float] = None
    debug: bool = False
    curv_stats_csv: Optional[str] = None


# ======================
# LAS I/O
# ======================
def load_points_from_las(path: str) -> Tuple[np.ndarray, laspy.LasData]:
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    return pts, las


# ======================
# 新增函数：为每个簇生成唯一颜色
# ======================
def generate_segment_colors(segment_local: np.ndarray) -> np.ndarray:
    """为每个 segment_id 生成随机 RGB 颜色"""
    n = len(segment_local)
    colors = np.zeros((n, 3), dtype=np.uint16)  # uint16 because LAS stores RGB as 16-bit

    mask = segment_local >= 0
    unique_ids = np.unique(segment_local[mask])

    rng = np.random.default_rng(42)  # 固定随机种子保证结果一致
    color_map = {
        sid: rng.integers(low=50, high=255, size=3, dtype=np.uint16) * 256 for sid in unique_ids
    }

    for sid, rgb in color_map.items():
        colors[segment_local == sid] = rgb

    return colors


# ======================
# 写出 LAS （支持颜色）
# ======================
# ======================
# 写出 LAS （支持颜色 + 删除未分类点）
# ======================
def write_main_las_with_segments(
    template_las: laspy.LasData,
    out_path: str,
    segment_local: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    # ✅ 1. 仅保留已分配点
    mask = segment_local >= 0
    n_removed = np.sum(~mask)
    print(f"[INFO] 已移除未分类点 {n_removed} 个 / 总点 {len(segment_local)}")

    # 过滤点与颜色
    filtered_points = template_las.points[mask].copy()
    filtered_segments = segment_local[mask]
    filtered_colors = colors[mask] if colors is not None else None

    # ✅ 2. 构造新 las
    las = laspy.LasData(template_las.header)
    las.points = filtered_points

    # 添加 segment 相关字段
    if "segment_id" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.uint32))
    if "segment_class" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="segment_class", type=np.uint16))
    if "segment_local" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="segment_local", type=np.uint32))

    seg_id = filtered_segments.astype(np.uint32)
    seg_cls = np.zeros_like(filtered_segments, dtype=np.uint16)
    seg_loc = filtered_segments.astype(np.uint32)

    las["segment_id"] = seg_id
    las["segment_class"] = seg_cls
    las["segment_local"] = seg_loc

    # ✅ 3. 写入颜色字段
    if filtered_colors is not None:
        if "red" not in las.point_format.dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="red", type=np.uint16))
            las.add_extra_dim(laspy.ExtraBytesParams(name="green", type=np.uint16))
            las.add_extra_dim(laspy.ExtraBytesParams(name="blue", type=np.uint16))
        las.red = filtered_colors[:, 0]
        las.green = filtered_colors[:, 1]
        las.blue = filtered_colors[:, 2]

    # ✅ 4. 写出
    las.write(out_path)
    print(f"[OK] 已写入分类后 LAS（未分类点已删除）：{out_path}")



# ======================
# 法向 & 曲率估计
# ======================
def estimate_normals_and_curvature(pts: np.ndarray, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    n_pts = len(pts)
    k_use = min(max(3, k), n_pts)
    nbrs = NearestNeighbors(n_neighbors=k_use, algorithm='auto').fit(pts)
    indices = nbrs.kneighbors(return_distance=False)

    normals = np.zeros_like(pts, dtype=np.float64)
    curv = np.zeros((n_pts,), dtype=np.float64)

    for i in range(n_pts):
        neigh = pts[indices[i]]
        c = neigh.mean(axis=0)
        X = neigh - c
        C = (X.T @ X) / max(1, len(neigh))
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)
        n = evecs[:, order[0]]
        if np.dot(n, pts[i] - c) > 0:
            n = -n
        normals[i] = n / (np.linalg.norm(n) + 1e-12)
        lam = evals[order]
        curv[i] = lam[0] / (lam.sum() + 1e-12)

    return normals, curv


# ======================
# 区域生长
# ======================
def region_grow_all(
    pts: np.ndarray,
    normals: np.ndarray,
    curv: np.ndarray,
    params: RGParams
) -> np.ndarray:
    n = pts.shape[0]
    if n == 0:
        return np.full((0,), -1, dtype=int)

    k_use = min(max(3, params.k_neighbors), n)
    nbrs = NearestNeighbors(n_neighbors=k_use, algorithm='auto').fit(pts)
    knn_idx = nbrs.kneighbors(return_distance=False)

    seed_thresh = np.percentile(curv, params.seed_curv_percentile)
    seeds = np.where(curv <= seed_thresh)[0]
    if seeds.size == 0:
        seeds = np.arange(n)

    if params.curv_auto_percentile is not None:
        curv_auto = float(np.percentile(curv, params.curv_auto_percentile))
        curv_thresh = min(params.curv_thresh, curv_auto)
    else:
        curv_thresh = params.curv_thresh

    assigned = np.full((n,), -1, dtype=int)
    local_cluster_id = 0
    cos_thresh = np.cos(np.deg2rad(params.theta_deg))

    for s0 in seeds:
        if assigned[s0] != -1:
            continue
        n_seed = normals[s0] / (np.linalg.norm(normals[s0]) + 1e-12)
        q = deque([s0]); cluster = []

        while q:
            u = q.popleft()
            if assigned[u] != -1:
                continue
            n_ref = n_seed if params.use_seed_normal else normals[u] / (np.linalg.norm(normals[u]) + 1e-12)
            if np.dot(normals[u], n_ref) < cos_thresh or curv[u] > curv_thresh:
                continue
            assigned[u] = local_cluster_id
            cluster.append(u)
            for v in knn_idx[u]:
                if assigned[v] != -1:
                    continue
                nv = normals[v] / (np.linalg.norm(normals[v]) + 1e-12)
                if np.dot(nv, n_ref) >= cos_thresh and curv[v] <= curv_thresh:
                    q.append(v)

        if len(cluster) < params.min_cluster_size:
            for u in cluster:
                assigned[u] = -1
        else:
            local_cluster_id += 1

    return assigned


# ======================
# 连续编号
# ======================
def remap_segment_ids(segment_local: np.ndarray) -> np.ndarray:
    mask = segment_local >= 0
    unique_old = np.unique(segment_local[mask])
    mapping = {old: new_id for new_id, old in enumerate(unique_old, start=1)}

    new_segment_local = segment_local.copy()
    for old, new in mapping.items():
        new_segment_local[segment_local == old] = new

    return new_segment_local


# ======================
# 统计
# ======================
def summarize_segments(segment_local: np.ndarray) -> Dict[str, int]:
    n_pts = int(segment_local.size)
    n_assigned = int(np.sum(segment_local >= 0))
    uniq = np.unique(segment_local[segment_local >= 0])
    return {"num_segments": int(len(uniq)), "num_points": n_pts, "num_assigned": n_assigned}


def print_summary(summary: Dict[str, int]):
    print(f"\n[Summary] segments={summary['num_segments']} | points={summary['num_points']} | assigned={summary['num_assigned']} | assign_ratio={summary['num_assigned']/summary['num_points']:.3f}\n")


# ======================
# 主函数
# ======================
# ======================
# 主函数（严格按 RGParams 默认参数运行）
# ======================
def main():
    # ====== 1. 手动指定输入输出路径（保持灵活） ======
    in_las  = r"D:\laz152\69-30\69-25\struct_plane_4.las"
    out_las = r"D:\laz152\69-30\69-25\fenge_colored4.las"

    # ====== 2. 使用 RGParams 中的默认参数 ======
    params = RGParams()  # ✅ 直接使用类默认值

    # 打印当前使用的参数以确认
    print("=== 使用 RGParams 默认参数 ===")
    print(f"k_neighbors={params.k_neighbors}")
    print(f"theta_deg={params.theta_deg}")
    print(f"curv_thresh={params.curv_thresh}")
    print(f"seed_curv_percentile={params.seed_curv_percentile}")
    print(f"min_cluster_size={params.min_cluster_size}")
    print("==============================")

    # ====== 3. 执行主逻辑 ======
    pts, las = load_points_from_las(in_las)
    normals, curv = estimate_normals_and_curvature(pts, k=params.k_neighbors)
    seg_local = region_grow_all(pts, normals, curv, params)
    seg_local = remap_segment_ids(seg_local)

    colors = generate_segment_colors(seg_local)
    summary = summarize_segments(seg_local)
    print_summary(summary)

    write_main_las_with_segments(las, out_las, seg_local, colors)
    print(f"[OK] 主 LAS 已写入并上色：{out_las}")


if __name__ == "__main__":
    main()

    
