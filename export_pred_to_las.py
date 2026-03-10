# export_pred_to_las.py
"""
把预测标签写回 LAS/LAZ：
- 输入：原始 las 文件 + 对应的 *_pred.npy（形状= (num_points,)）
- 输出：带 extra bytes 字段 pred_label 的新 las 文件

用途：
- 你想在 CloudCompare / PDAL / 自己的可视化工具里直接看预测结果
"""

import os
import argparse
import numpy as np
import laspy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_las", type=str, required=True, help="path to input .las/.laz")
    parser.add_argument("--pred_npy", type=str, required=True, help="path to *_pred.npy (num_points,)")
    parser.add_argument("--out_las", type=str, required=True, help="path to output .las/.laz")
    parser.add_argument("--field_name", type=str, default="pred_label", help="extra bytes field name")
    args = parser.parse_args()

    las = laspy.read(args.in_las)
    pred = np.load(args.pred_npy).astype(np.uint16)  # 类别数一般不大，用 uint16 足够

    # 注意：pred 的长度通常是你采样后的 num_points，不一定等于原始点数
    # 为了能写回，我们创建一个全长度数组，然后把前 len(pred) 写进去（仅用于快速可视化）
    # 如果你要对“原始所有点”写回，需要做全点推理，不能只采样。
    N = len(las.x)
    out_arr = np.zeros((N,), dtype=np.uint16)
    L = min(N, pred.shape[0])
    out_arr[:L] = pred[:L]

    # 若该字段不存在，新增 ExtraBytes
    dims = set(las.point_format.dimension_names)
    if args.field_name not in dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name=args.field_name, type=np.uint16))

    setattr(las, args.field_name, out_arr)

    # 写出
    las.write(args.out_las)
    print(f"Saved: {args.out_las}")

if __name__ == "__main__":
    main()