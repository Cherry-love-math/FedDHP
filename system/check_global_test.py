import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def describe(name, x):
    x = np.asarray(x)
    print(f"\n[{name}]")
    print("  dtype:", x.dtype)
    print("  shape:", x.shape)
    print("  min/max:", float(x.min()), float(x.max()))
    print("  mean/std:", float(x.mean()), float(x.std()))
    if x.ndim == 4:
        # 粗略看每通道均值方差（假设 NCHW 或 NHWC 都试一下）
        if x.shape[1] == 3:
            ch_mean = x.mean(axis=(0,2,3))
            ch_std  = x.std(axis=(0,2,3))
            print("  assume NCHW: per-channel mean:", ch_mean, "std:", ch_std)
        if x.shape[-1] == 3:
            ch_mean = x.mean(axis=(0,1,2))
            ch_std  = x.std(axis=(0,1,2))
            print("  assume NHWC: per-channel mean:", ch_mean, "std:", ch_std)

def assert_global_npz_ok(npz_path):
    assert os.path.exists(npz_path), f"NOT FOUND: {npz_path}"
    data = np.load(npz_path)
    assert "x" in data and "y" in data, "npz must contain keys: x, y"

    x = data["x"]
    y = data["y"]

    describe("global_test.x", x)
    describe("global_test.y", y)

    # 1) 标签检查
    assert y.ndim == 1, f"y should be 1D, got {y.ndim}D"
    assert y.dtype in (np.int64, np.int32), f"y dtype should be int, got {y.dtype}"
    assert x.shape[0] == y.shape[0], f"N mismatch: x has {x.shape[0]}, y has {y.shape[0]}"
    assert x.shape[0] == 10000, f"CIFAR-10 official test size should be 10000, got {x.shape[0]}"

    # 2) 形状/通道顺序检查
    assert x.ndim == 4, f"x should be 4D, got {x.ndim}D"
    is_nchw = (x.shape[1] == 3 and x.shape[2] == 32 and x.shape[3] == 32)
    is_nhwc = (x.shape[3] == 3 and x.shape[1] == 32 and x.shape[2] == 32)
    assert is_nchw or is_nhwc, f"x must be NCHW(10000,3,32,32) or NHWC(10000,32,32,3), got {x.shape}"

    # 3) 数值范围检查：应该是 normalize 后大约 [-1,1]
    xmin, xmax = float(x.min()), float(x.max())
    # 允许少量超界（例如插值/数值误差），但不应该是 [0,255]
    assert xmax <= 5.0 and xmin >= -5.0, (
        f"Range looks wrong. Expected ~[-1,1] (or within [-5,5]). Got [{xmin},{xmax}]"
    )
    assert not (xmin >= 0.0 and xmax > 10.0), (
        f"Looks like raw pixels [0,255] not normalized. Got [{xmin},{xmax}]"
    )

    # 4) 二次归一化的典型症状：范围被压到接近 [-3,3] 之外或均值/方差异常极端
    # 这里做一个启发式：如果 std 很小或很大，都可疑
    xstd = float(x.std())
    assert 0.1 < xstd < 2.0, f"Suspicious std={xstd}. Possible double-normalization or wrong format."

    return x, y, ("NCHW" if is_nchw else "NHWC")

def load_official_cifar10_test(transformed=True):
    if transformed:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    else:
        tf = transforms.Compose([transforms.ToTensor()])

    ds = torchvision.datasets.CIFAR10(root="../dataset/Cifar10_noniid_a0.1_wg/rawdata", train=False, download=False, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs, 0).numpy().astype(np.float32, copy=False)  # NCHW
    y = torch.cat(ys, 0).numpy().astype(np.int64, copy=False)
    return x, y

def compare_distribution(a, b, name_a="A", name_b="B"):
    a = np.asarray(a).astype(np.float32, copy=False)
    b = np.asarray(b).astype(np.float32, copy=False)
    # 统一成 NCHW 再对比
    def to_nchw(x):
        if x.shape[1] == 3:
            return x
        return np.transpose(x, (0,3,1,2))
    a = to_nchw(a)
    b = to_nchw(b)

    # 抽样对比，避免输出太大
    idx = np.random.RandomState(0).choice(a.shape[0], size=256, replace=False)
    a_s = a[idx]
    b_s = b[idx]

    ma, sa = float(a_s.mean()), float(a_s.std())
    mb, sb = float(b_s.mean()), float(b_s.std())
    print(f"\n[DistCompare] {name_a} vs {name_b}")
    print(f"  {name_a}: mean={ma:.4f} std={sa:.4f}")
    print(f"  {name_b}: mean={mb:.4f} std={sb:.4f}")
    print(f"  mean diff={abs(ma-mb):.4f}  std diff={abs(sa-sb):.4f}")

    # 经验阈值：同一预处理管线下差异不应很大
    assert abs(ma-mb) < 0.2 and abs(sa-sb) < 0.2, (
        "Distribution mismatch is large. Possible format mismatch or normalization mismatch."
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_global_test.py <path_to_global_test.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    xg, yg, fmt = assert_global_npz_ok(npz_path)
    print(f"\n[OK] global_test.npz basic checks passed. Format detected: {fmt}")

    # 与“官方 test + transform”对齐验证
    xo, yo = load_official_cifar10_test(transformed=True)
    describe("official_test_transformed.x", xo)
    compare_distribution(xg, xo, "global_test", "official_test_transformed")

    # 再额外对比“官方 test 未 normalize”的范围，帮助你确认是否二次/零次归一化
    xr, yr = load_official_cifar10_test(transformed=False)
    describe("official_test_raw_toTensor.x", xr)

    print("\n[ALL GOOD] No obvious format mismatch / double-normalization detected.")

if __name__ == "__main__":
    main()
