import numpy as np
import os
import sys
import random
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import glob
import hashlib

random.seed(1)
np.random.seed(1)
num_clients = 20

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

def _to_3ch_32(x):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.size(2) != 32 or x.size(3) != 32:
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
    return x

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    global_test_path = os.path.join(dir_path, "global_test.npz")
    global_calib_path = os.path.join(dir_path, "global_calib.npz")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition) and os.path.exists(global_test_path) and os.path.exists(global_calib_path):
        return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=dir_path + "rawdata", train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    train_x = trainset.data
    train_y = trainset.targets
    test_x = testset.data
    test_y = testset.targets

    train_x = _to_3ch_32(train_x)
    test_x = _to_3ch_32(test_x)

    x_global = test_x.cpu().detach().numpy().astype(np.float32, copy=False)
    y_global = test_y.cpu().detach().numpy().astype(np.int64, copy=False)
    np.savez_compressed(global_test_path, x=x_global, y=y_global)
    print(f"[GlobalTest] saved to: {global_test_path}, x={x_global.shape}, y={y_global.shape}")

    x_train_all = train_x.cpu().detach().numpy().astype(np.float32, copy=False)
    y_train_all = train_y.cpu().detach().numpy().astype(np.int64, copy=False)

    N_CALIB = 2000
    rng = np.random.RandomState(1)
    idx = rng.choice(len(x_train_all), size=min(N_CALIB, len(x_train_all)), replace=False)
    x_calib = x_train_all[idx]
    y_calib = y_train_all[idx]
    np.savez_compressed(global_calib_path, x=x_calib, y=y_calib)
    print(f"[GlobalCalib] saved to: {global_calib_path}, x={x_calib.shape}, y={y_calib.shape}")

    dataset_image = x_train_all
    dataset_label = y_train_all

    num_classes = int(np.unique(dataset_label).shape[0])
    print(f"Number of classes: {num_classes}")

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    outdir = sys.argv[4] if len(sys.argv) > 4 else "FashionMNIST"

    dir_path = os.path.join(os.path.dirname(__file__), outdir + "/")
    generate_dataset(dir_path, num_clients, niid, balance, partition)
    verify_no_global_leak(dir_path)