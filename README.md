# FedDHP

Official implementation of **FedDHP: Dual-Head Prior-Aware Distillation for Global Generalization and Personalized Adaptation in Federated Learning**.

FedDHP is a federated learning framework designed for heterogeneous client data distributions. It jointly considers:

- **G-FL**: generic global-model generalization;
- **P-FL**: personalized client-side adaptation;
- **communication efficiency** under compressed collaborative parameter exchange.

This repository is built upon [PFLlib](https://github.com/TsingZ0/PFLlib), an open-source personalized federated learning library released under the Apache License 2.0. We sincerely thank the PFLlib authors for their valuable framework.

---

## 1. Main Features

Compared with the original PFLlib framework, this repository adds or adapts the following components for the FedDHP paper:

- FedDHP dual-head generic-personalized training framework;
- prior-aware logit adjustment under heterogeneous label priors;
- asymmetric weak/strong augmentation for branch-role separation;
- calibrated knowledge distillation and feature alignment;
- dual-criterion evaluation protocol:
  - global generic model evaluation;
  - client-side personalized model evaluation;
- classifier-level fine-tuning evaluation for applicable methods;
- communication-cost measurement;
- low-rank communication compression support;
- adapted baseline evaluation under the same G-FL/P-FL protocol;
- representative scripts and configurations for reproducing the main experiments.

---

## 2. Relationship with PFLlib

This repository is **not a from-scratch federated learning library**. It is adapted from PFLlib.

The following parts are inherited from or adapted based on PFLlib:

- client/server training framework;
- dataset generation utilities;
- baseline algorithm implementations;
- model definitions and training utilities;
- simulation infrastructure.

The following parts are added or substantially modified for FedDHP:

- FedDHP client/server logic;
- dual-head training;
- prior-aware logit adjustment;
- asymmetric-view training;
- knowledge distillation and feature alignment;
- G-FL/P-FL dual evaluation;
- classifier-level fine-tuning evaluation;
- communication-cost calculation;
- experiment scripts for the FedDHP paper.

Unless otherwise specified, the core algorithmic logic of baseline methods follows the PFLlib implementation. Some baseline interfaces were adapted to support the unified evaluation protocol used in our paper.

---

## 3. Repository Structure

```text
FedDHP/
├── README.md
├── LICENSE
├── NOTICE
├── environment.yml
├── requirements.txt
├── dataset/
│   ├── README.md
│   ├── generate_Cifar10.py
│   ├── generate_Cifar100.py
│   ├── generate_FashionMNIST.py
│   └── utils/
├── system/
│   ├── main.py
│   ├── flcore/
│   │   ├── clients/
│   │   ├── servers/
│   │   ├── trainmodel/
│   │   └── optimizers/
│   └── utils/
├── scripts/
│   ├── prepare_data.sh
│   ├── run_feddhp_cifar10.sh
│   └── run_feddhp_strong_aug.sh
└── configs/
    └── paper_settings.md
```

The FedDHP implementation is mainly located in:

```text
system/flcore/clients/
system/flcore/servers/
```

---

## 4. Environment

We recommend using Conda.

```bash
conda env create -f environment.yml
conda activate feddhp
```

Alternatively, install the minimal pip dependencies:

```bash
pip install -r requirements.txt
```

The experiments were conducted with PyTorch 2.0.1 and CUDA 11.8. If PyTorch installation fails, please install PyTorch following the official PyTorch instructions for your CUDA version.

### AutoDL / GLIBCXX compatibility note

On AutoDL or similar Linux servers, if you encounter `GLIBCXX` or `libstdc++` compatibility errors, try:

```bash
conda install -y -c conda-forge libstdcxx-ng "gcc_impl_linux-64>=12" "gxx_linux-64>=12"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
```

This step is not always required. Use it only when your environment reports related library compatibility errors.

---

## 5. Dataset Preparation

Raw datasets and generated client partitions are **not included** in this repository due to file size limitations.

Please generate federated datasets locally.

```bash
cd dataset
```

### CIFAR-10

```bash
python generate_Cifar10.py noniid - dir Cifar10_noniid_a0.1
python generate_Cifar10.py noniid - dir Cifar10_noniid_a0.3
```

### CIFAR-100

```bash
python generate_Cifar100.py noniid - dir Cifar100_noniid_a0.1
python generate_Cifar100.py noniid - dir Cifar100_noniid_a0.3
```

### Fashion-MNIST

```bash
python generate_FashionMNIST.py noniid - dir FashionMNIST_noniid_a0.1
python generate_FashionMNIST.py noniid - dir FashionMNIST_noniid_a0.3
```

Then return to the project root:

```bash
cd ..
```

The main experiments use Dirichlet label-skew partitions with concentration parameters:

```text
alpha = 0.1
alpha = 0.3
```

A smaller alpha indicates stronger label-distribution heterogeneity.

---

## 6. Quick Start

### FedDHP on CIFAR-10 / Dir(0.1) / ResNet10

```bash
bash scripts/feddhp/run_table1_cifar10_dir01_resnet10.sh
```

Equivalent command:

```bash
python main.py \
  -dev cuda \
  -did 0 \
  -data Cifar10_noniid_a0.1_wg2 \
  -ncl 10 \
  -m ResNet10 \
  -algo FedDHP \
  -nc 20 \
  -jr 1.0 \
  -lbs 64 \
  -ls 3 \
  -lr 0.15 \
  -mlr 0.1 \
  -gr 300 \
  -eg 5 \
  -asd_beta 4.0 \
  -yoyo_tau 2.0 \
  -yoyo_gamma 0.5 \
  -asd_gamma 1.0 \
  -Ts 0.99 \
  -Te 0.99
```

This is a representative main-table setting for FedDHP on CIFAR-10 under Dirichlet non-IID partition with `alpha = 0.1`.

---

## 7. Strong Augmentation Example

```bash
bash scripts/feddhp/run_strong_aug_cifar10_dir01.sh
```

Equivalent command:

```bash
python main.py \
  -dev cuda \
  -did 0 \
  -data Cifar10_noniid_a0.1 \
  -ncl 10 \
  -m ResNet10 \
  -algo FedDHP \
  -nc 20 \
  -jr 1.0 \
  -lbs 64 \
  -ls 3 \
  -lr 0.15 \
  -mlr 0.1 \
  -gr 300 \
  -eg 5 \
  -asd_beta 4.0 \
  -yoyo_tau 2.0 \
  -yoyo_gamma 0.5 \
  -asd_gamma 1.0 \
  -Ts 0.99 \
  -Te 0.99 \
  --student_aug crop_hflip_cutout \
  --cifar_crop_padding 4 \
  --cifar_cutout_p 0.5 \
  -go aug_crop_hflip_cutout
```

---

## 8. Ablation Example

### Without strong augmentation

```bash
bash scripts/feddhp/run_ablation_no_strong_aug_cifar10_dir01.sh
```

Equivalent command:

```bash
python main.py \
  -dev cuda \
  -did 0 \
  -data Cifar10_noniid_a0.1 \
  -ncl 10 \
  -m ResNet10 \
  -algo FedDHP \
  -nc 20 \
  -jr 1.0 \
  -lbs 64 \
  -ls 3 \
  -lr 0.15 \
  -mlr 0.1 \
  -gr 300 \
  -eg 5 \
  -asd_beta 4.0 \
  -yoyo_tau 2.0 \
  -yoyo_gamma 0.5 \
  -asd_gamma 1.0 \
  -Ts 0.99 \
  -Te 0.99 \
  --ablate_strong_aug
```

---
## 9. Main Experimental Settings

The main experiments follow the settings below:

```text
Datasets: CIFAR-10, CIFAR-100, Fashion-MNIST
Partition: Dirichlet label skew
Dirichlet alpha: 0.1 and 0.3
Number of clients: 20
Client participation ratio: 1.0
Global rounds: 300
Local epochs: 3
Default backbone: ResNet10
Additional backbones: ResNet18 and VGG11-BN
Optimizer: SGD

Batch size:
  CIFAR-10: 64
  Fashion-MNIST: 64
  CIFAR-100: 32

Heterogeneity:
  Dirichlet alpha = 0.1
  Dirichlet alpha = 0.3
```

To change the dataset or backbone, modify:

```bash
-data Cifar10_noniid_a0.1
-m ResNet10
```

Examples:

```bash
-data Cifar100_noniid_a0.1
-data FashionMNIST_noniid_a0.1
-m ResNet18
-m VGG11_BN
```

Please check the exact model names supported in `system/main.py` and `system/flcore/trainmodel/`.

---

## 10. Baselines

The paper compares FedDHP with nine representative baselines.

### Generic FL

- FedAvg
- FedProx
- FedGen

### Personalized FL

- FedPAC
- pFedMe
- FedBABU

### Balanced / Generic-Personalized FL

- FedKD
- FedRoD
- GPFL

The baseline implementations are inherited from or adapted based on PFLlib. In this repository, we modified the evaluation pipeline to support:

- G-FL evaluation;
- P-FL evaluation;
- classifier-level fine-tuning evaluation;
- communication-cost calculation;
- unified reporting under the same experimental protocol.

Representative baseline scripts are provided under:

```text
scripts/baselines/
```

Additional baseline scripts can be added following the same command format.

---

## 11. Evaluation Protocol

FedDHP adopts a dual-criterion evaluation protocol.

### G-FL: Generic Evaluation

The aggregated global model is evaluated on the unified global test set.

### P-FL: Personalized Evaluation

The deployable client-side model is evaluated on each client's local test set. The final result is reported as the test-sample-weighted average accuracy across clients.

For applicable methods, classifier-level fine-tuning is applied before P-FL evaluation. For methods with native personalized evaluation pipelines, we follow their corresponding evaluation protocol.

---

## 12. Communication Cost

FedDHP includes communication-cost measurement and low-rank compression support.

Communication cost is calculated based on transmitted collaborative parameters and compression-related settings. The personalized head remains local and is not uploaded to the server.

Please refer to the FedDHP client/server implementation for detailed behavior.

---

## 13. Reproducibility Notes

Federated learning experiments can be affected by:

- random data partitioning;
- random initialization;
- client sampling;
- GPU and CUDA environment;
- PyTorch/CUDA versions;
- local training randomness;
- algorithm-specific hyperparameters.

Therefore, reproduced numbers may show slight variation from the reported results in the paper.

For closer reproduction, please carefully match:

- dataset partition;
- random seed;
- backbone;
- optimizer;
- local epochs;
- global rounds;
- batch size;
- learning rate;
- algorithm-specific hyperparameters;
- CUDA/PyTorch environment.

The provided scripts are intended to reproduce the main experimental settings and implementation logic used in the paper.

---

## 14. Notes on Unused PFLlib Algorithms

PFLlib supports many FL and pFL algorithms. This repository focuses on FedDHP and the baselines used in the FedDHP paper.

Some inherited PFLlib modules may remain in the codebase to preserve compatibility with the original framework. However, only the methods listed in the baseline section are used for the main comparisons in our paper.

---

## 15. License

This repository is released under the Apache License 2.0.

Since this repository is built upon PFLlib, we retain the Apache License 2.0 and provide attribution to the original PFLlib project.

Please see:

```text
LICENSE
NOTICE
```

---

## 16. Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{chen2026feddhp,
  title={FedDHP: Dual-head prior-aware distillation for global generalization and personalized adaptation in federated learning},
  journal={Expert Systems with Applications},
  year={2026}
}
```

Please also cite PFLlib if you use the underlying framework:

```bibtex
@article{zhang2025pfllib,
  title={PFLlib: A Beginner-Friendly and Comprehensive Personalized Federated Learning Library and Benchmark},
  author={Zhang, Jianqing and Liu, Yang and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Cao, Jian},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={50},
  pages={1--10},
  year={2025}
}
```

---

## 17. Acknowledgement

This repository is built upon PFLlib. We sincerely thank the PFLlib authors for releasing their federated learning library and benchmark.

Portions of the baseline algorithm implementations, client/server framework, dataset generation utilities, and training infrastructure are inherited from or adapted based on PFLlib. FedDHP-specific training logic, dual-criterion evaluation, classifier-level fine-tuning evaluation, communication-cost measurement, and experimental scripts are added in this repository.