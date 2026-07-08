import numpy as np
import os
import sys
import random
import glob
import hashlib
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
def relabel(c):
    if isinstance(c, (int, np.integer)):
        v = int(c)
        if 0 <= v < 62:
            return v
        c = str(v)
    s = str(c)
    v = int(s, 16)
    if 0x30 <= v <= 0x39:
        return v - 0x30
    if 0x41 <= v <= 0x5A:
        return v - 0x41 + 10
    if 0x61 <= v <= 0x7A:
        return v - 0x61 + 36
    raise ValueError(s)

def _safe_check(config_path, train_path, test_path, num_clients, niid, balance, partition, num_classes):
    try:
        return check(config_path, train_path, test_path, num_clients, niid, balance, partition)
    except TypeError:
        return check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition)
def _load_meta(meta_path):
    return pd.read_pickle(meta_path)
def _rows_by_writer(meta):
    d = {}
    for wid, rows in meta:
        d[wid] = rows
    return d
def _get_writer_ids(meta, num_clients, prefer_top=True, rng=None):
    images_per_writer = [(row[0], len(row[1])) for row in meta]
    images_per_writer.sort(key=lambda x: x[1], reverse=True)
    if prefer_top:
        return [w for w, _ in images_per_writer[:num_clients]]
    if rng is None:
        rng = np.random.RandomState(1)
    ids = [w for w, _ in images_per_writer]
    if len(ids) <= num_clients:
        return ids
    idx = rng.choice(len(ids), size=num_clients, replace=False)
    return [ids[i] for i in idx]
def _resolve_image_path(rel_path, extra_root):
    p1 = os.path.join(ROOT_PATH, rel_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(ROOT_PATH, extra_root, rel_path)
    if os.path.exists(p2):
        return p2
    return p1
def _load_rows_as_numpy(rows, transform, extra_root):
    xs = []
    ys = []
    for rel_path, lab in rows:
        path = _resolve_image_path(rel_path, extra_root)
        img = Image.open(path).convert("L")
        t = transform(img)
     #   if t.dim() == 3 and t.size(0) == 1:
     #       t = t.repeat(3, 1, 1)
        xs.append(t)
        ys.append(relabel(lab))
    if len(xs) == 0:
        x = np.empty((0, 1, 28, 28), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        return x, y
   
    x = torch.stack(xs, dim=0).cpu().numpy().astype(np.float32, copy=False)
    y = np.array(ys, dtype=np.int64)
    return x, y
def _hash_img(arr):
    a = np.ascontiguousarray(arr)
    return hashlib.sha1(a.tobytes()).digest()
def _load_xy_from_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    if "x" in z and "y" in z:
        return z["x"], z["y"]
    if "data" in z:
        d = z["data"].item()
        return d["x"], d["y"]
    raise KeyError(f"unknown npz format: {npz_path}")
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
def _save_global_npz(path, x, y, tag):
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)
    np.savez_compressed(path, x=x, y=y)
    print(f"[{tag}] saved to: {path}, x={x.shape}, y={y.shape}")
def _stat_from_labels(labels):
    if labels.size == 0:
        return []
    cls, cnt = np.unique(labels, return_counts=True)
    return [(int(c), int(n)) for c, n in zip(cls.tolist(), cnt.tolist())]
def _extract_train_pool(train_data, fallback_X, fallback_y):
    try:
        xs = []
        ys = []
        if isinstance(train_data, dict):
            keys = sorted(list(train_data.keys()))
            for k in keys:
                d = train_data[k]
                xs.append(d["x"])
                ys.append(d["y"])
        else:
            for d in train_data:
                if isinstance(d, dict) and "x" in d and "y" in d:
                    xs.append(d["x"])
                    ys.append(d["y"])
                elif isinstance(d, (list, tuple)) and len(d) == 2:
                    xs.append(d[0])
                    ys.append(d[1])
        x_pool = np.concatenate(xs, axis=0) if len(xs) else np.empty((0, 1, 28, 28), dtype=np.float32)
        y_pool = np.concatenate(ys, axis=0) if len(ys) else np.empty((0,), dtype=np.int64)
        return x_pool, y_pool
    except Exception:
        x_pool = np.concatenate(fallback_X, axis=0) if len(fallback_X) else np.empty((0, 3, 28, 28), dtype=np.float32)
        y_pool = np.concatenate(fallback_y, axis=0) if len(fallback_y) else np.empty((0,), dtype=np.int64)
        return x_pool, y_pool
def generate_dataset(dir_path, femnist_root, meta_file_name, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    num_classes = 62
    if _safe_check(config_path, train_path, test_path, num_clients, niid, balance, partition, num_classes):
        verify_no_global_leak(dir_path)
        return
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    meta_path = os.path.join(femnist_root, "intermediate", meta_file_name)
    if not os.path.exists(meta_path):
        meta_path = os.path.join(femnist_root, "data", "intermediate", meta_file_name)

    meta = _load_meta(meta_path)
    by_writer = _rows_by_writer(meta)
    writer_ids = _get_writer_ids(meta, num_clients, prefer_top=True, rng=np.random.RandomState(1))
    remaining_ids = [wid for wid in by_writer.keys() if wid not in set(writer_ids)]
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    for i, wid in enumerate(writer_ids):
        rows = by_writer[wid]
        xi, yi = _load_rows_as_numpy(rows, transform, femnist_root)
        X[i] = xi
        y[i] = yi
        statistic[i] = _stat_from_labels(yi)
    if (partition is None) or (partition == "writer"):
        if not niid:
            all_x = np.concatenate([X[i] for i in range(num_clients)], axis=0)
            all_y = np.concatenate([y[i] for i in range(num_clients)], axis=0)
            X, y, statistic = separate_data((all_x, all_y), num_clients, num_classes, niid, balance, None, class_per_client=2)
    else:
        all_x = np.concatenate([X[i] for i in range(num_clients)], axis=0)
        all_y = np.concatenate([y[i] for i in range(num_clients)], axis=0)
        X, y, statistic = separate_data((all_x, all_y), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    global_test_path = os.path.join(dir_path, "global_test.npz")
    if len(remaining_ids) > 0:
        gx_list = []
        gy_list = []
        total = 0
        for wid in remaining_ids:
            rows = by_writer[wid]
            xi, yi = _load_rows_as_numpy(rows, transform, femnist_root)
            if xi.shape[0] > 0:
                gx_list.append(xi)
                gy_list.append(yi)
                total += xi.shape[0]
            if total >= 10000:
                break
        if len(gx_list) > 0:
            gx = np.concatenate(gx_list, axis=0)
            gy = np.concatenate(gy_list, axis=0)
            if gx.shape[0] > 10000:
                rng = np.random.RandomState(1)
                idx = rng.choice(gx.shape[0], size=10000, replace=False)
                gx = gx[idx]
                gy = gy[idx]
            _save_global_npz(global_test_path, gx, gy, "GlobalTest")
        else:
            _save_global_npz(global_test_path, np.empty((0, 3, 28, 28), dtype=np.float32), np.empty((0,), dtype=np.int64), "GlobalTest")
    else:
        all_x = np.concatenate([X[i] for i in range(num_clients)], axis=0)
        all_y = np.concatenate([y[i] for i in range(num_clients)], axis=0)
        if all_x.shape[0] > 0:
            rng = np.random.RandomState(1)
            n_gt = min(10000, max(1, all_x.shape[0] // 10))
            idx = rng.choice(all_x.shape[0], size=n_gt, replace=False)
            gx = all_x[idx]
            gy = all_y[idx]
            gset = set(_hash_img(gx[i]) for i in range(gx.shape[0]))
            for i in range(num_clients):
                keep = [j for j in range(X[i].shape[0]) if _hash_img(X[i][j]) not in gset]
                X[i] = X[i][keep]
                y[i] = y[i][keep]
                statistic[i] = _stat_from_labels(y[i])
            _save_global_npz(global_test_path, gx, gy, "GlobalTest")
        else:
            _save_global_npz(global_test_path, np.empty((0, 3, 28, 28), dtype=np.float32), np.empty((0,), dtype=np.int64), "GlobalTest")
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, None if partition == "writer" else partition)
    global_calib_path = os.path.join(dir_path, "global_calib.npz")
    N_CALIB = 2000
    x_pool, y_pool = _extract_train_pool(train_data, X, y)
    if x_pool.shape[0] > 0:
        rng = np.random.RandomState(1)
        idx = rng.choice(x_pool.shape[0], size=min(N_CALIB, x_pool.shape[0]), replace=False)
        x_calib = x_pool[idx].astype(np.float32, copy=False)
        y_calib = y_pool[idx].astype(np.int64, copy=False)
        np.savez_compressed(global_calib_path, x=x_calib, y=y_calib)
        print(f"[GlobalCalib] saved to: {global_calib_path}, x={x_calib.shape}, y={y_calib.shape}")
    else:
        np.savez_compressed(global_calib_path, x=x_pool.astype(np.float32, copy=False), y=y_pool.astype(np.int64, copy=False))
        print(f"[GlobalCalib] saved to: {global_calib_path}, x={x_pool.shape}, y={y_pool.shape}")
    verify_no_global_leak(dir_path)
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    niid = True if len(sys.argv) > 1 and sys.argv[1] == "noniid" else False
    balance = True if len(sys.argv) > 2 and sys.argv[2] == "balance" else False
    partition = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    outdir = sys.argv[4] if len(sys.argv) > 4 else "FEMNIST"
    num_clients = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    femnist_root = sys.argv[6] if len(sys.argv) > 6 else "FEMNIST"
    meta_file_name = sys.argv[7] if len(sys.argv) > 7 else "images_by_writer.pkl"
    dir_path = os.path.join(os.path.dirname(__file__), outdir + "/")
    generate_dataset(dir_path, femnist_root, meta_file_name, num_clients, niid, balance, partition)
