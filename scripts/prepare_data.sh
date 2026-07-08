#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/dataset"

python generate_Cifar10.py noniid - dir Cifar10_noniid_a0.1
python generate_Cifar10.py noniid - dir Cifar10_noniid_a0.3

python generate_Cifar100.py noniid - dir Cifar100_noniid_a0.1
python generate_Cifar100.py noniid - dir Cifar100_noniid_a0.3

python generate_FashionMNIST.py noniid - dir FashionMNIST_noniid_a0.1
python generate_FashionMNIST.py noniid - dir FashionMNIST_noniid_a0.3
