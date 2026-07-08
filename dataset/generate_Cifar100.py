import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import glob
import hashlib

# 设置随机种子
random.seed(1)
np.random.seed(1)
num_clients = 20
# dir_path 会在 main 中根据参数动态设置

# --------------------------------------------------------------------------
# 新增功能：验证函数 (防止 Global Test 数据泄露到客户端)
# --------------------------------------------------------------------------
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
    # 建立全局测试集的哈希集合
    gset = set(_hash_img(gx[i]) for i in range(gx.shape[0]))
    
    total_client = 0
    leak_total = 0
    
    # 检查 train 和 test 文件夹下的所有客户端数据
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

# --------------------------------------------------------------------------
# 数据生成主函数
# --------------------------------------------------------------------------
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
        
    # Get Cifar100 data
    # 使用与 Cifar10 相同的 transform 逻辑
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    
    # 使用 DataLoader 一次性加载并应用 transform
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    # -------------------------------------------------------------------
    # 新增功能 1: 保存全局测试集 global_test.npz
    # -------------------------------------------------------------------
    global_test_path = os.path.join(dir_path, "global_test.npz")
    
    # 提取测试集数据并转为 numpy
    x_global = testset.data.cpu().detach().numpy()
    y_global = testset.targets.cpu().detach().numpy()
    
    # 显式转换类型
    x_global = x_global.astype(np.float32, copy=False)
    y_global = y_global.astype(np.int64, copy=False)
    
    np.savez_compressed(global_test_path, x=x_global, y=y_global)
    print(f"[GlobalTest] saved to: {global_test_path}, x={x_global.shape}, y={y_global.shape}")

    # -------------------------------------------------------------------
    # 新增功能 2: 保存全局校准集 global_calib.npz (从 Train 中采样)
    # -------------------------------------------------------------------
    global_calib_path = os.path.join(dir_path, "global_calib.npz")
    N_CALIB = 2000 # 可以根据 CIFAR-100 总量适当调整，这里保持 2000

    x_train_all = trainset.data.cpu().detach().numpy().astype(np.float32, copy=False)
    y_train_all = trainset.targets.cpu().detach().numpy().astype(np.int64, copy=False)

    rng = np.random.RandomState(1)
    idx = rng.choice(len(x_train_all), size=min(N_CALIB, len(x_train_all)), replace=False)

    x_calib = x_train_all[idx]
    y_calib = y_train_all[idx]

    np.savez_compressed(global_calib_path, x=x_calib, y=y_calib)
    print(f"[GlobalCalib] saved to: {global_calib_path}, x={x_calib.shape}, y={y_calib.shape}")

    # -------------------------------------------------------------------
    # 构建客户端数据集 (Dataset Partitioning)
    # -------------------------------------------------------------------
    dataset_image = []
    dataset_label = []

    # [关键修改] 
    # 为了防止数据泄露并匹配 Cifar10 代码逻辑，只将 Trainset 分给客户端。
    # 原始 Cifar100 代码合并了 Testset，这里注释掉以保持一致性。
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    # dataset_image.extend(testset.data.cpu().detach().numpy()) 
    
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    # dataset_label.extend(testset.targets.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # Cifar100 类别多，class_per_client 保持为 10 是合理的
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=10)
    
    train_data, test_data = split_data(X, y)
    
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    # 解析命令行参数
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    
    # 新增：支持自定义输出目录名称
    outdir = sys.argv[4] if len(sys.argv) > 4 else "Cifar100"
    dir_path = os.path.join(os.path.dirname(__file__), outdir + "/")

    generate_dataset(dir_path, num_clients, niid, balance, partition)
    
    # 执行泄露检查
    verify_no_global_leak(dir_path)