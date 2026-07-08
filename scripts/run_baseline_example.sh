#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/system"

python main.py \
  -dev cuda \
  -did 0 \
  -data Cifar10_noniid_a0.1 \
  -ncl 10 \
  -m ResNet10 \
  -algo FedAvg \
  -nc 20 \
  -jr 1.0 \
  -lbs 64 \
  -ls 3 \
  -lr 0.1 \
  -gr 300 \
  -eg 5
