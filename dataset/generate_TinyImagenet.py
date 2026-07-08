import os
import sys
import random
import hashlib
import glob
import zipfile
import urllib.request
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)

num_clients = 20
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_NAME = "tiny-imagenet-200.zip"
DATA_DIR_NAME = "tiny-imagenet-200"
N_CALIB = 2000
CLASS_PER_CLIENT = 20

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        self.classes = imagefolder_obj.classes
        self.class_to_idx = imagefolder_obj.class_to_idx
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples, dtype=object)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples, dtype=object)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = int(self.samples[index][1])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples) if self.dataidxs is None else len(self.dataidxs)

class TinyImageNetVal(torch.utils.data.Dataset):
    def __init__(self, val_root, wnid_to_idx, transform=None):
        self.val_root = val_root
        self.transform = transform
        anno = os.path.join(val_root, "val_annotations.txt")
        img_dir = os.path.join(val_root, "images")
        samples = []
        with open(anno, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in wnid_to_idx:
                    continue
                samples.append((os.path.join(img_dir, img_name), int(wnid_to_idx[wnid])))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)

def _download(url, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    urllib.request.urlretrieve(url, dst_path)

def _unzip(zip_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)

def _ensure_rawdata(dir_path):
    raw_root = os.path.join(dir_path, "rawdata")
    zip_path = os.path.join(raw_root, ZIP_NAME)
    extracted = os.path.join(raw_root, DATA_DIR_NAME)
    if not os.path.exists(extracted):
        _download(URL, zip_path)
        _unzip(zip_path, raw_root)
    return extracted

def _load_all_to_tensor(dataset, batch_size=256, num_workers=2):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    n = len(dataset)
    x = torch.empty((n, 3, 64, 64), dtype=torch.float32)
    y = torch.empty((n,), dtype=torch.int64)
    p = 0
    for xb, yb in loader:
        bs = xb.size(0)
        x[p:p+bs].copy_(xb)
        y[p:p+bs].copy_(yb.to(torch.int64))
        p += bs
    return x, y

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
    os.makedirs(dir_path, exist_ok=True)
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train") + os.sep
    test_path = os.path.join(dir_path, "test") + os.sep

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    extracted = _ensure_rawdata(dir_path)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_root = os.path.join(extracted, "train")
    val_root = os.path.join(extracted, "val")

    trainset = ImageFolder_custom(root=train_root, transform=transform)
    wnid_to_idx = trainset.class_to_idx

    valset = TinyImageNetVal(val_root=val_root, wnid_to_idx=wnid_to_idx, transform=transform)

    x_train_t, y_train_t = _load_all_to_tensor(trainset, batch_size=256, num_workers=2)
    x_val_t, y_val_t = _load_all_to_tensor(valset, batch_size=256, num_workers=2)

    x_global = x_val_t.numpy().astype(np.float32, copy=False)
    y_global = y_val_t.numpy().astype(np.int64, copy=False)
    global_test_path = os.path.join(dir_path, "global_test.npz")
    np.savez_compressed(global_test_path, x=x_global, y=y_global)
    print(f"[GlobalTest] saved to: {global_test_path}, x={x_global.shape}, y={y_global.shape}")

    x_train_all = x_train_t.numpy().astype(np.float32, copy=False)
    y_train_all = y_train_t.numpy().astype(np.int64, copy=False)
    rng = np.random.RandomState(1)
    idx = rng.choice(len(x_train_all), size=min(N_CALIB, len(x_train_all)), replace=False)
    x_calib = x_train_all[idx]
    y_calib = y_train_all[idx]
    global_calib_path = os.path.join(dir_path, "global_calib.npz")
    np.savez_compressed(global_calib_path, x=x_calib, y=y_calib)
    print(f"[GlobalCalib] saved to: {global_calib_path}, x={x_calib.shape}, y={y_calib.shape}")

    dataset_image = x_train_all
    dataset_label = y_train_all
    num_classes = int(len(set(dataset_label.tolist())))
    print(f"Number of classes: {num_classes}")

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=CLASS_PER_CLIENT
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
    outdir = sys.argv[4] if len(sys.argv) > 4 else "TinyImagenet"

    dir_path = os.path.join(os.path.dirname(__file__), outdir) + os.sep

    generate_dataset(dir_path, num_clients, niid, balance, partition)
    verify_no_global_leak(dir_path)
