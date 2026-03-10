import torch
from models_pointnet2 import PointNet2SemSeg

def quick_test():
    print("Running quick test for PointNet++ rock discontinuity model...")

    # 你的模型要求 xyz 形状为 (B, N, 3)
    B, N = 1, 2048
    xyz = torch.rand(B, N, 3).float()

    # 这里只做最简单的 quick test：只输入 xyz，不加额外特征
    # 因此 feature_dim = 0，features = None
    model = PointNet2SemSeg(num_classes=3, feature_dim=0)
    model.eval()

    with torch.no_grad():
        logits = model(xyz, None)

    print("Quick test passed!")
    print("Input xyz shape :", xyz.shape)
    print("Output logits shape:", logits.shape)

    # 额外做一个简单检查
    assert logits.shape == (B, N, 3), f"Unexpected output shape: {logits.shape}"

if __name__ == "__main__":
    quick_test()