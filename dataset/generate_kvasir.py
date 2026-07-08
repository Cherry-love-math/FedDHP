import sys
import pandas as pd
import numpy as np
import os
import random
import glob
import hashlib
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from utils.dataset_utils import check, separate_data, split_data, save_file, ImageDataset
from torch.utils.data import DataLoader

random.seed(1)
np.random.seed(1)

num_clients = 20
img_size = 112
data_path = os.path.join(os.path.dirname(__file__), "kvasir/")

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

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    data_dir = os.path.join(data_path, "rawdata/kvasir-dataset-v2/")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Raw Kvasir data not found: {data_dir}")

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    print("All class names:", class_names)

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    file_names = []
    labels = []

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp"]
    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for file_name in sorted(os.listdir(cls_dir)):
            if os.path.splitext(file_name)[1].lower() in valid_ext:
                file_names.append(os.path.join(cls, file_name))
                labels.append(label)

    df = pd.DataFrame({"file_name": file_names, "class": labels})
    print("Total raw images:", len(df))

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageDataset(df, data_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    x_list = []
    y_list = []

    for xb, yb in dataloader:
        x_list.append(xb.cpu().numpy().astype(np.float32, copy=False))
        y_list.append(yb.cpu().numpy().astype(np.int64, copy=False))

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print("Loaded data:", x_all.shape, y_all.shape)

    idx_all = np.arange(len(y_all))
    train_idx, global_test_idx = train_test_split(
        idx_all,
        test_size=0.2,
        random_state=1,
        stratify=y_all
    )

    x_global = x_all[global_test_idx]
    y_global = y_all[global_test_idx]

    global_test_path = os.path.join(dir_path, "global_test.npz")
    np.savez_compressed(global_test_path, x=x_global, y=y_global)
    print(f"[GlobalTest] saved to: {global_test_path}, x={x_global.shape}, y={y_global.shape}")

    x_train_pool = x_all[train_idx]
    y_train_pool = y_all[train_idx]

    global_calib_path = os.path.join(dir_path, "global_calib.npz")
    N_CALIB = 1000
    rng = np.random.RandomState(1)
    calib_idx = rng.choice(len(x_train_pool), size=min(N_CALIB, len(x_train_pool)), replace=False)

    x_calib = x_train_pool[calib_idx]
    y_calib = y_train_pool[calib_idx]

    np.savez_compressed(global_calib_path, x=x_calib, y=y_calib)
    print(f"[GlobalCalib] saved to: {global_calib_path}, x={x_calib.shape}, y={y_calib.shape}")

    X, y, statistic = separate_data(
        (x_train_pool, y_train_pool),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=2
    )

    train_data, test_data = split_data(X, y)

    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition
    )

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    outdir = sys.argv[4] if len(sys.argv) > 4 else "kvasir_noniid_a0.1"

    dir_path = os.path.join(os.path.dirname(__file__), outdir + "/")

    generate_dataset(dir_path, num_clients, niid, balance, partition)
    verify_no_global_leak(dir_path)