import os
import sys
import glob
import hashlib
import random
import numpy as np
import torchvision
from utils.dataset_utils import check, separate_data, split_data, save_file

def _patch_emnist_url():
    try:
        from torchvision.datasets import EMNIST
        EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
        EMNIST.md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    except Exception:
        pass

def _load_xy_from_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    if "x" in z and "y" in z:
        return z["x"], z["y"]
    if "data" in z:
        d = z["data"].item()
        return d["x"], d["y"]
    raise KeyError(f"unknown npz format: {npz_path}")

def _hash_img(arr):
    a = np.ascontiguousarray(arr)
    return hashlib.sha1(a.tobytes()).digest()

def verify_no_global_leak(dataset_dir):
    gpath = os.path.join(dataset_dir, "global_test.npz")
    if not os.path.exists(gpath):
        print(f"[LeakCheck][SKIP] global_test not found: {gpath}")
        return
    gx, gy = _load_xy_from_npz(gpath)
    gset = set(_hash_img(gx[i]) for i in range(gx.shape[0]))
    total_client = 0
    leak_total = 0
    for split in ["train", "test"]:
        files = sorted(glob.glob(os.path.join(dataset_dir, split, "*.npz")))
        for p in files:
            x, y = _load_xy_from_npz(p)
            total_client += int(x.shape[0])
            c = 0
            for i in range(x.shape[0]):
                if _hash_img(x[i]) in gset:
                    c += 1
            if c:
                leak_total += c
                print(f"[LeakCheck][LEAK] {split}/{os.path.basename(p)} leaked={c}")
    print(f"[LeakCheck] global_test={gx.shape[0]} client_total(train+test)={total_client}")
    if leak_total == 0:
        print("[LeakCheck][OK] no global_test leakage into any client train/test")
    else:
        print(f"[LeakCheck][BAD] leaked_samples={leak_total}")

def _postprocess_x_uint8(x_u8, make_3ch=True, fix_orientation=True, pad32=True):
    x = x_u8
    if x.ndim == 3:
        x = x[:, None, :, :]
    if fix_orientation:
        x = np.rot90(x, k=-1, axes=(2, 3))
        x = np.flip(x, axis=2)
    if pad32 and x.shape[2] == 28 and x.shape[3] == 28:
        x = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0)
    x = x.astype(np.float32, copy=False) / 255.0
    x = x * 2.0 - 1.0
    if make_3ch and x.shape[1] == 1:
        x = np.repeat(x, 3, axis=1)
    return x.astype(np.float32, copy=False)

def _apply_postprocess_to_split(split_data_obj, make_3ch=True, fix_orientation=True, pad32=True):
    if isinstance(split_data_obj, dict):
        for k in list(split_data_obj.keys()):
            d = split_data_obj[k]
            d["x"] = _postprocess_x_uint8(d["x"], make_3ch=make_3ch, fix_orientation=fix_orientation, pad32=pad32)
            d["y"] = np.asarray(d["y"], dtype=np.int64)
            split_data_obj[k] = d
        return split_data_obj
    for i in range(len(split_data_obj)):
        d = split_data_obj[i]
        if isinstance(d, dict) and "x" in d and "y" in d:
            d["x"] = _postprocess_x_uint8(d["x"], make_3ch=make_3ch, fix_orientation=fix_orientation, pad32=pad32)
            d["y"] = np.asarray(d["y"], dtype=np.int64)
            split_data_obj[i] = d
        elif isinstance(d, (list, tuple)) and len(d) == 2:
            x, y = d
            x = _postprocess_x_uint8(x, make_3ch=make_3ch, fix_orientation=fix_orientation, pad32=pad32)
            y = np.asarray(y, dtype=np.int64)
            split_data_obj[i] = (x, y)
    return split_data_obj

def _save_global_npz(path, x, y, tag):
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)
    np.savez_compressed(path, x=x, y=y)
    print(f"[{tag}] saved to: {path}, x={x.shape}, y={y.shape}")

def _try_load_emnist(raw_root, split_name, train_flag):
    from torchvision.datasets import EMNIST
    try:
        ds = EMNIST(root=raw_root, split=split_name, train=train_flag, download=False, transform=None)
        return ds
    except RuntimeError:
        ds = EMNIST(root=raw_root, split=split_name, train=train_flag, download=True, transform=None)
        return ds

def generate_dataset(dir_path, num_clients, niid, balance, partition, split_name="digits",
                     make_3ch=True, fix_orientation=True, pad32=True, n_calib=2000, max_global_test=10000):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        verify_no_global_leak(dir_path)
        return
    _patch_emnist_url()
    raw_root = os.path.join(dir_path, "rawdata")
    trainset = _try_load_emnist(raw_root, split_name, True)
    testset = _try_load_emnist(raw_root, split_name, False)
    x_train_u8 = trainset.data.numpy()
    y_train = trainset.targets.numpy()
    x_test_u8 = testset.data.numpy()
    y_test = testset.targets.numpy()
    x_train_u8 = x_train_u8[:, None, :, :]
    x_test_u8 = x_test_u8[:, None, :, :]
    num_classes = int(len(set(y_train.tolist())))
    print(f"[EMNIST] split={split_name} train={x_train_u8.shape[0]} test={x_test_u8.shape[0]} num_classes={num_classes}")
    global_test_path = os.path.join(dir_path, "global_test.npz")
    if max_global_test is not None and int(max_global_test) > 0 and x_test_u8.shape[0] > int(max_global_test):
        rng = np.random.RandomState(1)
        idx = rng.choice(x_test_u8.shape[0], size=int(max_global_test), replace=False)
        gx_u8 = x_test_u8[idx]
        gy = y_test[idx]
    else:
        gx_u8 = x_test_u8
        gy = y_test
    gx = _postprocess_x_uint8(gx_u8, make_3ch=make_3ch, fix_orientation=fix_orientation, pad32=pad32)
    _save_global_npz(global_test_path, gx, gy, "GlobalTest")
    global_calib_path = os.path.join(dir_path, "global_calib.npz")
    if x_train_u8.shape[0] > 0:
        rng = np.random.RandomState(1)
        idx = rng.choice(x_train_u8.shape[0], size=min(int(n_calib), x_train_u8.shape[0]), replace=False)
        cx_u8 = x_train_u8[idx]
        cy = y_train[idx]
    else:
        cx_u8 = x_train_u8
        cy = y_train
    cx = _postprocess_x_uint8(cx_u8, make_3ch=make_3ch, fix_orientation=fix_orientation, pad32=pad32)
    _save_global_npz(global_calib_path, cx, cy, "GlobalCalib")
    dataset_image = x_train_u8
    dataset_label = y_train
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    train_data = _apply_postprocess_to_split(train_data, make_3ch=make_3ch, fix_orientation=fix_orientation, pad32=pad32)
    test_data = _apply_postprocess_to_split(test_data, make_3ch=make_3ch, fix_orientation=fix_orientation, pad32=pad32)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)
    verify_no_global_leak(dir_path)

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    niid = True if len(sys.argv) > 1 and sys.argv[1] == "noniid" else False
    balance = True if len(sys.argv) > 2 and sys.argv[2] == "balance" else False
    partition = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    outdir = sys.argv[4] if len(sys.argv) > 4 else "EMNIST_digits"
    num_clients = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    split_name = sys.argv[6] if len(sys.argv) > 6 else "digits"
    dir_path = os.path.join(os.path.dirname(__file__), outdir + "/")
    generate_dataset(dir_path, num_clients, niid, balance, partition, split_name=split_name,
                     make_3ch=True, fix_orientation=True, pad32=True, n_calib=2000, max_global_test=10000)
