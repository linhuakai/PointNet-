import os
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
import laspy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_las_seg import (
    LASPointSegDataset,
    _read_xyz,
    _read_labels,
    _normalize_unit_sphere,
    remap_labels,
)
from models_pointnet2 import PointNet2SemSeg


def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int):
    """
    计算 mIoU（语义分割常用指标）
    pred, gt: (N,)
    """
    per_class = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        inter = np.logical_and(pred == c, gt == c).sum()
        union = np.logical_or(pred == c, gt == c).sum()
        per_class[c] = np.nan if union == 0 else inter / union
    miou = float(np.nanmean(per_class)) if np.any(~np.isnan(per_class)) else 0.0
    return miou, per_class


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total_correct = 0
    total_seen = 0
    miou_list = []

    for xyz, feats, labels, _ in tqdm(loader, desc="Val/Test", leave=False):
        xyz = xyz.to(device)              # (B,P,3)
        labels = labels.to(device)        # (B,P)
        feats = feats.to(device) if feats is not None else None

        logits = model(xyz, feats)        # (B,P,C)
        pred = torch.argmax(logits, dim=-1)

        total_correct += (pred == labels).sum().item()
        total_seen += labels.numel()

        p_np = pred.reshape(-1).cpu().numpy()
        g_np = labels.reshape(-1).cpu().numpy()
        miou, _ = compute_iou(p_np, g_np, num_classes)
        miou_list.append(miou)

    oa = total_correct / (total_seen + 1e-9)
    miou = float(np.mean(miou_list)) if len(miou_list) else 0.0
    return oa, miou


def parse_feature_names(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def infer_feature_dim(feature_names: list[str]) -> int:
    d = 0
    for n in feature_names:
        d += 3 if n == "rgb" else 1
    return d


def compute_class_weights_from_train(train_loader, num_classes: int, device):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, _, labels, _ in tqdm(train_loader, desc="CountLabels", leave=False):
        lab = labels.numpy().reshape(-1)
        for c in range(num_classes):
            counts[c] += np.sum(lab == c)

    freq = counts / (counts.sum() + 1e-9)
    weights = 1.0 / np.log(1.02 + freq + 1e-9)
    weights = weights / (weights.mean() + 1e-9)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_features_from_las(las: laspy.LasData, feature_names: list[str]) -> np.ndarray | None:
    """
    和 dataset 的逻辑保持一致，直接从原始 LAS 构建输入特征。
    返回: (N, Din) 或 None
    """
    if len(feature_names) == 0:
        return None

    dims = set(las.point_format.dimension_names)
    feats_list = []

    for name in feature_names:
        if name == "rgb":
            if not {"red", "green", "blue"}.issubset(dims):
                raise KeyError("You requested feature 'rgb' but LAS has no red/green/blue.")
            rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32)
            rgb = rgb / 65535.0
            feats_list.append(rgb)

        elif name == "intensity":
            if "intensity" not in dims:
                raise KeyError("LAS has no 'intensity'")
            x = np.asarray(las.intensity).astype(np.float32) / 65535.0
            feats_list.append(x[:, None])

        elif name == "return_number":
            if "return_number" not in dims:
                raise KeyError("LAS has no 'return_number'")
            x = np.asarray(las.return_number).astype(np.float32) / 8.0
            feats_list.append(x[:, None])

        elif name == "num_returns":
            if "number_of_returns" not in dims:
                raise KeyError("LAS has no 'number_of_returns'")
            x = np.asarray(las.number_of_returns).astype(np.float32) / 8.0
            feats_list.append(x[:, None])

        elif name == "classification":
            if "classification" not in dims:
                raise KeyError("LAS has no 'classification'")
            x = np.asarray(las.classification).astype(np.float32) / 255.0
            feats_list.append(x[:, None])

        else:
            raise ValueError(f"Unknown feature name: {name}")

    return np.concatenate(feats_list, axis=1).astype(np.float32)


@torch.no_grad()
def predict_full_cloud_with_voting(
    model,
    device,
    xyz_full: np.ndarray,
    feats_full: np.ndarray | None,
    num_points: int,
    num_classes: int,
    vote_rounds: int = 1,
):
    """
    对整幅点云做“分块 + 多轮投票”预测。
    输出:
        pred_full: (N,) 连续类别 id
        vote_count: (N,) 每个点被投票次数
    说明:
    - 每一轮先打乱所有点，再按 num_points 分块
    - 最后一块不足 num_points 时，用该块内部重复采样补齐
    - 因此每一轮都保证所有原始点至少被覆盖一次
    """
    model.eval()

    N = xyz_full.shape[0]
    vote_sum = np.zeros((N, num_classes), dtype=np.float32)
    vote_count = np.zeros((N,), dtype=np.int32)

    for _ in range(vote_rounds):
        perm = np.random.permutation(N)

        for start in range(0, N, num_points):
            idx_chunk = perm[start:start + num_points]

            if idx_chunk.shape[0] < num_points:
                pad = np.random.choice(idx_chunk, size=(num_points - idx_chunk.shape[0]), replace=True)
                choice = np.concatenate([idx_chunk, pad], axis=0)
            else:
                choice = idx_chunk

            xyz_chunk = xyz_full[choice].copy()
            xyz_chunk = _normalize_unit_sphere(xyz_chunk)

            if feats_full is not None:
                feats_chunk = feats_full[choice].copy()
                feats_t = torch.from_numpy(feats_chunk).float().unsqueeze(0).to(device)   # (1,P,D)
            else:
                feats_t = None

            xyz_t = torch.from_numpy(xyz_chunk).float().unsqueeze(0).to(device)           # (1,P,3)

            logits = model(xyz_t, feats_t)                                                # (1,P,C)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()                # (P,C)

            # 对 choice 中每个位置累加概率投票
            np.add.at(vote_sum, choice, probs)
            np.add.at(vote_count, choice, 1)

    pred_full = np.argmax(vote_sum, axis=1).astype(np.int64)
    return pred_full, vote_count


def save_full_prediction_to_las(
    in_las_path: str,
    out_las_path: str,
    pred_new: np.ndarray,
    vote_count: np.ndarray,
    new_to_raw: dict[int, int] | None = None,
):
    """
    把整云预测写回原始 LAS：
    - 坐标、点数、原始字段全部保持不变
    - classification 尽量写 raw label（若 <=255）
    - 额外写 pred_label(连续id)、vote_count
    """
    las = laspy.read(in_las_path)
    N = len(las.x)

    assert pred_new.shape[0] == N
    assert vote_count.shape[0] == N

    dims = set(las.point_format.dimension_names)

    pred_new_u16 = pred_new.astype(np.uint16)
    vote_count_u16 = np.clip(vote_count, 0, 65535).astype(np.uint16)

    if new_to_raw is not None:
        pred_raw = np.vectorize(lambda x: new_to_raw[int(x)])(pred_new).astype(np.int64)
    else:
        pred_raw = pred_new.astype(np.int64)

    # classification 是标准字段，尽量写 raw label（若范围允许）
    if pred_raw.max() <= 255 and pred_raw.min() >= 0:
        las.classification = pred_raw.astype(np.uint8)
    else:
        las.classification = np.clip(pred_new, 0, 255).astype(np.uint8)

    if "pred_label" not in dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name="pred_label", type=np.uint16))
    las.pred_label = pred_new_u16

    if "vote_count" not in dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name="vote_count", type=np.uint16))
    las.vote_count = vote_count_u16

    if "pred_raw" not in dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name="pred_raw", type=np.uint16))
    las.pred_raw = np.clip(pred_raw, 0, 65535).astype(np.uint16)

    las.write(out_las_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs/pointnet2_las_seg")

    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--feature_names", type=str, default="intensity", help="comma-separated, e.g. intensity,return_number,num_returns,rgb")
    parser.add_argument("--use_class_weight", action="store_true", help="use class weights computed from train set")
    parser.add_argument("--save_pred_dir", type=str, default="", help="in test mode, save sampled pred/gt npy per file")
    parser.add_argument("--save_las_dir", type=str, default="", help="in test mode, save full-cloud predicted LAS")
    parser.add_argument("--vote_rounds", type=int, default=1, help="number of full-cloud voting rounds when exporting LAS")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_names = parse_feature_names(args.feature_names)
    feature_dim = infer_feature_dim(feature_names)

    mapping_path = os.path.join(args.out_dir, "label_mapping.json")

    if args.mode == "train":
        train_set = LASPointSegDataset(
            root_dir=args.data_root,
            split="train",
            num_points=args.num_points,
            feature_names=feature_names,
            augment=True,
            cache_label_mapping_path=mapping_path,
        )

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)
        raw_to_new = {int(k): int(v) for k, v in mapping_json["raw_to_new"].items()}
        num_classes = int(mapping_json["num_classes"])

        val_set = LASPointSegDataset(
            root_dir=args.data_root,
            split="val",
            num_points=args.num_points,
            feature_names=feature_names,
            augment=False,
            label_mapping=raw_to_new,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )

        model = PointNet2SemSeg(num_classes=num_classes, feature_dim=feature_dim).to(device)

        if args.use_class_weight:
            class_w = compute_class_weights_from_train(train_loader, num_classes, device)
            criterion = nn.CrossEntropyLoss(weight=class_w)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        start_epoch = 1
        best_miou = -1.0

        if args.ckpt and os.path.isfile(args.ckpt):
            ckpt = torch.load(args.ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optim"])
            scheduler.load_state_dict(ckpt["sched"])
            start_epoch = ckpt["epoch"] + 1
            best_miou = ckpt.get("best_miou", best_miou)
            print(f"[Resume] epoch={start_epoch}, best_miou={best_miou:.4f}")

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")

            running_loss = 0.0
            step_count = 0

            for xyz, feats, labels, _ in pbar:
                xyz = xyz.to(device)
                labels = labels.to(device)
                feats = feats.to(device) if feats is not None else None

                optimizer.zero_grad(set_to_none=True)

                logits = model(xyz, feats)
                loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step_count += 1
                pbar.set_postfix(
                    loss=running_loss / max(1, step_count),
                    lr=optimizer.param_groups[0]["lr"]
                )

            scheduler.step()

            val_oa, val_miou = evaluate(model, val_loader, device, num_classes)
            print(f"[Epoch {epoch}] val_OA={val_oa:.4f}, val_mIoU={val_miou:.4f}")

            last_path = os.path.join(args.out_dir, "ckpt_last.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "best_miou": best_miou,
                "num_classes": num_classes,
                "feature_dim": feature_dim,
                "feature_names": feature_names,
            }, last_path)

            if val_miou > best_miou:
                best_miou = val_miou
                best_path = os.path.join(args.out_dir, "ckpt_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_miou": best_miou,
                    "num_classes": num_classes,
                    "feature_dim": feature_dim,
                    "feature_names": feature_names,
                }, best_path)
                print(f"[Best] saved: {best_path} (mIoU={best_miou:.4f})")

    else:
        assert os.path.isfile(mapping_path), f"Missing {mapping_path}. Train first to generate label mapping."
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)

        raw_to_new = {int(k): int(v) for k, v in mapping_json["raw_to_new"].items()}
        new_to_raw = {int(v): int(k) for k, v in raw_to_new.items()}
        num_classes = int(mapping_json["num_classes"])

        assert args.ckpt and os.path.isfile(args.ckpt), "In test mode, --ckpt must be a valid checkpoint path."
        ckpt = torch.load(args.ckpt, map_location="cpu")

        ckpt_feature_names = ckpt.get("feature_names", feature_names)
        ckpt_feature_dim = ckpt.get("feature_dim", feature_dim)
        feature_names = ckpt_feature_names
        feature_dim = ckpt_feature_dim

        model = PointNet2SemSeg(num_classes=num_classes, feature_dim=feature_dim).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        test_set = LASPointSegDataset(
            root_dir=args.data_root,
            split="test",
            num_points=args.num_points,
            feature_names=feature_names,
            augment=False,
            label_mapping=raw_to_new,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )

        oa, miou = evaluate(model, test_loader, device, num_classes)
        print(f"[Test] OA={oa:.4f}, mIoU={miou:.4f}")

        # 保留你原来采样点的 npy 导出
        if args.save_pred_dir:
            os.makedirs(args.save_pred_dir, exist_ok=True)
            with torch.no_grad():
                for xyz, feats, labels, fname in tqdm(test_loader, desc="SavePred"):
                    xyz = xyz.to(device)
                    feats = feats.to(device) if feats is not None else None
                    logits = model(xyz, feats)
                    pred = torch.argmax(logits, dim=-1).cpu().numpy()
                    gt = labels.numpy()

                    for b in range(pred.shape[0]):
                        base = os.path.splitext(fname[b])[0]
                        np.save(os.path.join(args.save_pred_dir, base + "_pred.npy"), pred[b])
                        np.save(os.path.join(args.save_pred_dir, base + "_gt.npy"), gt[b])

            print(f"[Test] Saved sampled predictions to: {args.save_pred_dir}")

        # 新增：整云导出
        if args.save_las_dir:
            os.makedirs(args.save_las_dir, exist_ok=True)

            test_files = test_set.files
            for in_path in tqdm(test_files, desc="ExportFullLAS"):
                las = laspy.read(in_path)

                xyz_full = _read_xyz(las)                              # (N,3)
                feats_full = build_features_from_las(las, feature_names)
                raw_labels_full = _read_labels(las).astype(np.int64)
                _ = remap_labels(raw_labels_full, raw_to_new)          # 这里只是确保映射可用

                pred_full, vote_count = predict_full_cloud_with_voting(
                    model=model,
                    device=device,
                    xyz_full=xyz_full,
                    feats_full=feats_full,
                    num_points=args.num_points,
                    num_classes=num_classes,
                    vote_rounds=max(1, args.vote_rounds),
                )

                base = os.path.splitext(os.path.basename(in_path))[0]
                out_path = os.path.join(args.save_las_dir, base + "_pred.las")

                save_full_prediction_to_las(
                    in_las_path=in_path,
                    out_las_path=out_path,
                    pred_new=pred_full,
                    vote_count=vote_count,
                    new_to_raw=new_to_raw,
                )

            print(f"[Test] Saved full-cloud LAS files to: {args.save_las_dir}")


if __name__ == "__main__":
    main()