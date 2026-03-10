# pointnet2_utils.py
"""
PointNet++ 的核心几何操作（纯 PyTorch 实现）：
- FPS: farthest point sampling
- Ball query: 局部邻域分组
- Feature interpolation: 特征传播 FP 使用的 3-NN 插值

输入/输出张量维度在注释里写得很清楚，方便你调试。
"""

import torch

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    计算两组点的 pairwise squared distance。
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M)
    """
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))  # (B,N,M)
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)   # (B,N,1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)    # (B,1,M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    用 idx 从 points 里取点（支持 2D / 3D idx）。
    points: (B, N, C)
    idx:    (B, S) 或 (B, S, K)
    return: (B, S, C) 或 (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)       # (B,1,1,...)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1                                # (1,S,K,...)

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    FPS 采样：从 N 个点中采样 npoint 个“尽量分散”的点作为中心点。
    xyz: (B, N, 3)
    return: centroids_idx (B, npoint)
    """
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)  # 每个点到已选中心点的最小距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)   # (B,1,3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)            # (B,N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]                  # 下一个最远点
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query：对每个中心点 new_xyz，在 xyz 里找半径 radius 内的点，最多取 nsample 个。
    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    return: group_idx (B, S, nsample)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)  # (B,S,N)
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius * radius] = N  # 超出半径的点标记为 N（无效）

    # 取前 nsample 个最小索引（因为无效点= N 会排到最后）
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # 如果一个中心点周围有效点不足 nsample，会出现 N，需要用第一个有效点填补
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor | None):
    """
    采样 + 分组：
    1) FPS 取 npoint 个中心点 new_xyz
    2) ball query 分组出每个中心点的 nsample 个邻域点
    3) 形成 new_points：拼接相对坐标 (grouped_xyz - new_xyz) 和原始特征 points

    xyz: (B, N, 3)
    points: (B, N, D) 或 None
    return:
      new_xyz: (B, npoint, 3)
      new_points: (B, npoint, nsample, 3 + D)  (包含相对坐标)
    """
    fps_idx = farthest_point_sample(xyz, npoint)         # (B,npoint)
    new_xyz = index_points(xyz, fps_idx)                 # (B,npoint,3)

    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B,npoint,nsample)
    grouped_xyz = index_points(xyz, idx)                   # (B,npoint,nsample,3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # 相对坐标

    if points is not None:
        grouped_points = index_points(points, idx)         # (B,npoint,nsample,D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points

def three_nn(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    FP用：对 xyz1 的每个点，在 xyz2 中找最近的 3 个点。
    xyz1: (B, N, 3)  需要插值的点（密集）
    xyz2: (B, S, 3)  已知特征的点（稀疏）
    return:
      dist: (B, N, 3)  三个近邻距离
      idx:  (B, N, 3)  三个近邻索引
    """
    dists = square_distance(xyz1, xyz2)  # (B,N,S)
    dist, idx = torch.topk(dists, k=3, dim=-1, largest=False, sorted=True)
    dist = torch.clamp(dist, min=1e-10)
    return torch.sqrt(dist), idx

def three_interpolate(points: torch.Tensor, idx: torch.Tensor, dist: torch.Tensor):
    """
    三近邻反距离加权插值
    points: (B, S, C)  xyz2 上的特征
    idx:    (B, N, 3)
    dist:   (B, N, 3)
    return: (B, N, C)
    """
    inv_dist = 1.0 / dist
    norm = torch.sum(inv_dist, dim=2, keepdim=True)
    weight = inv_dist / norm
    interpolated_points = torch.sum(index_points(points, idx) * weight.unsqueeze(-1), dim=2)
    return interpolated_points