import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
import torch
import numpy as np
import h5py
import copy
import time
import random
import json
from utils.data_utils import read_client_data
from utils.dlg import DLG
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from utils.metrics import accuracy_topk


class Server(object):
    def __init__(self, args, times):
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        # Set up the main attributes
        self.args = args
        self.seed = int(getattr(args, "seed", 3))
        self.deterministic = bool(getattr(args, "deterministic", False))
        self._set_global_seed(self.seed, deterministic=self.deterministic)
        self.eval_records = []
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
# --- mentor jitter tracking (for stability analysis / ablation) ---
        self.jitter_window = int(getattr(args, "jitter_window", 5))
        self._mentor_top1_hist = []
        self._mentor_top5_hist = []
        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new
        ts = time.strftime("%Y%m%d-%H%M%S")
        proj_root = Path(__file__).resolve().parents[1]   # 指向 PFLlib/system/ 的上一级
        self.logdir = proj_root / "runs" / f"{self.dataset}_{self.algorithm}_{ts}"
  #      self.logdir = proj_root / "runs" / f"{self.dataset}_{self.algorithm}_{ts}_ResNet18"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=self.logdir.as_posix())
        self.metrics_path = self.logdir / "metrics_eval.tsv"
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", encoding="utf-8") as f:          
              #   f.write("eval_idx\tglobal_round\tmentee_local_top1\tmentor_local_top1\tglobal_noBN_top1\tglobal_BN_top1\tglobal_BN_top5\tmentor_jitter_std\tmentor_jitter_TV\tmentor_jitter_TVmean\n")
              # #  f.write("eval_idx\tglobal_round\tmentee_local_top1\tmentor_local_top1\tglobal_noBN_top1\tglobal_BN_top1\tglobal_BN_top5\ttotal_uplink_MB\ttotal_downlink_MB\n")
                f.write("eval_idx\tglobal_round\tmentee_local_top1\tmentor_local_top1\tmentor_posthoc_local_top1\tmentor_posthoc_delta\tglobal_noBN_top1\tglobal_BN_top1\tglobal_BN_top5\n")
        self.summary_path = self.logdir / "summary.json"
        print("TensorBoard logdir:", self.logdir)  # 便于你确认路径
        self._init_global_calib_loader()
        self._init_global_test_loader()
    def _init_global_test_loader(self):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", self.dataset)
        dataset_dir = os.path.abspath(dataset_dir)

        path = os.path.join(dataset_dir, "global_test.npz")
        if not os.path.exists(path):
            print(f"[GlobalTest][WARN] not found: {path}. global test eval disabled.")
            self.global_test_loader = None
            return

        data = np.load(path)
        x = data["x"]
        y = data["y"]

        # x: (10000, 3, 32, 32) float32   y: (10000,) int64
        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).long()

        ds = TensorDataset(x_t, y_t)
        self.global_test_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        x, y = next(iter(self.global_test_loader))
        print(f"Global Test Data Range: min={x.min():.3f}, max={x.max():.3f}")
        # 正常应该是 -1 到 1 之间
        print(f"[GlobalTest] loaded: {path}, n={len(ds)}")
    def _init_global_calib_loader(self):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", self.dataset)
        dataset_dir = os.path.abspath(dataset_dir)

        path = os.path.join(dataset_dir, "global_calib.npz")
        if not os.path.exists(path):
            print(f"[GlobalCalib][WARN] not found: {path}. BN recalib disabled.")
            self.global_calib_loader = None
            return

        data = np.load(path)
        x = data["x"]
        y = data["y"]

        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).long()

        ds = TensorDataset(x_t, y_t)
        self.global_calib_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        xb, yb = next(iter(self.global_calib_loader))
        print(f"Global Calib Data Range: min={xb.min():.3f}, max={xb.max():.3f}")
        print(f"[GlobalCalib] loaded: {path}, n={len(ds)}")

    def _eval_global_test_with_model(self, model, bn_recalib=False):
        if self.global_test_loader is None:
            return None, None
      #      update_bn_stats(model, self.global_test_loader, self.device)
        if bn_recalib:
            if self.global_calib_loader is None:
                return None, None
            self.update_bn_stats(model, self.global_calib_loader, self.device)

        model.eval()
        top1_correct = 0.0
        top5_correct = 0.0
        total = 0.0

        with torch.no_grad():
            for x, y in self.global_test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)
                t1, t5 = accuracy_topk(logits, y, topk=(1, 5))
                b = y.size(0)

                top1_correct += float(t1) * b
                top5_correct += float(t5) * b
                total += b

        if total <= 0:
            return None, None

        return top1_correct / total, top5_correct / total
    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

#     def aggregate_parameters(self):


#         self.global_model = copy.deepcopy(self.uploaded_models[0])
#         for param in self.global_model.parameters():
#             param.data.zero_()
            
#         for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
#             self.add_parameters(w, client_model)
    def aggregate_parameters(self):
        if not self.uploaded_models:
            return

        # 1. 以第一个模型为模板，获取结构
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        global_sd = self.global_model.state_dict()

        # 2. 初始化累加器
        acc = {}
        for k, v in global_sd.items():
            if torch.is_floating_point(v):
                acc[k] = torch.zeros_like(v)
            else:
                # 对于整型 buffer (如 num_batches_tracked)，先复制第一个模型的作为基准
                acc[k] = v.clone()

        # 3. 开始加权平均
        # uploaded_weights 已经是归一化的 (sum=1)，所以直接乘
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            client_sd = client_model.state_dict()
            for k, v in client_sd.items():
                if k not in acc: continue # 防御性编程

                if torch.is_floating_point(v):
                    acc[k] += v * float(w)
                else:
                    # 整型处理：取最大值 (严谨写法)
                    # 记录训练过的总 batch 数，取最大值代表最“老”的经验
                    if 'num_batches_tracked' in k:
                        acc[k] = torch.max(acc[k], v)
                    else:
                        acc[k] = v 

        # 4. 加载回 Global Model
        self.global_model.load_state_dict(acc, strict=True)
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

#     def test_metrics(self):
#         if self.eval_new_clients and self.num_new_clients > 0:
#             self.fine_tuning_new_clients()
#             return self.test_metrics_new_clients()
        
#         num_samples = []
#         tot_correct = []
#         tot_auc = []
#         for c in self.clients:
#             ct, ns, auc = c.test_metrics()
#             tot_correct.append(ct*1.0)
#             tot_auc.append(auc*ns)
#             num_samples.append(ns)

#         ids = [c.id for c in self.clients]

#         return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

#     # evaluate selected clients
#     def evaluate(self, acc=None, loss=None):
#         stats = self.test_metrics()
#         stats_train = self.train_metrics()

#         test_acc = sum(stats[2])*1.0 / sum(stats[1])
#         test_auc = sum(stats[3])*1.0 / sum(stats[1])
#         train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
#         accs = [a / n for a, n in zip(stats[2], stats[1])]
#         aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
#         if acc == None:
#             self.rs_test_acc.append(test_acc)
#         else:
#             acc.append(test_acc)
        
#         if loss == None:
#             self.rs_train_loss.append(train_loss)
#         else:
#             loss.append(train_loss)

#         print("Averaged Train Loss: {:.4f}".format(train_loss))
#         print("Averaged Test Accuracy: {:.4f}".format(test_acc))
#         print("Averaged Test AUC: {:.4f}".format(test_auc))
#         # self.print_(test_acc, train_acc, train_loss)
#         print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
#         print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        
#         r = getattr(self, "current_round", len(self.rs_test_acc)-1)
#         self.tb.add_scalar("test/avg_acc",  test_acc,  r)
#         self.tb.add_scalar("test/avg_auc",  test_auc,  r)
#         self.tb.add_scalar("train/avg_loss",train_loss, r)
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        # mentee（学徒，用作主度量）与 mentor 的独立累计
        mentee_ns = mentee_top1 = mentee_top5 = mentee_auc_w = 0.0
        mentor_ns = mentor_top1 = mentor_top5 = mentor_auc_w = 0.0
        mentor_posthoc_ns = mentor_posthoc_top1 = mentor_posthoc_top5 = mentor_posthoc_auc_w = 0.0
        ids = []
        for c in self.clients:
            out = c.test_metrics()

            def add(role, tup):
                nonlocal mentee_ns, mentee_top1, mentee_top5, mentee_auc_w
                nonlocal mentor_ns, mentor_top1, mentor_top5, mentor_auc_w
                nonlocal mentor_posthoc_ns, mentor_posthoc_top1, mentor_posthoc_top5, mentor_posthoc_auc_w
                if len(tup) == 3:
                    ct, ns, auc = tup
                    top5 = ct  # 兼容无 top5 的老实现
                else:
                    ct, ns, auc, top5 = tup

                if role == "mentee":
                    mentee_ns   += ns
                    mentee_top1 += float(ct)
                    mentee_top5 += float(top5)
                    mentee_auc_w+= (auc if auc is not None else 0.0) * ns
                elif role == "mentor":
                    mentor_ns   += ns
                    mentor_top1 += float(ct)
                    mentor_top5 += float(top5)
                    mentor_auc_w+= (auc if auc is not None else 0.0) * ns
                elif role == "mentor_posthoc":
                    mentor_posthoc_ns += ns
                    mentor_posthoc_top1 += float(ct)
                    mentor_posthoc_top5 += float(top5)
                    mentor_posthoc_auc_w += (auc if auc is not None else 0.0) * ns
            if isinstance(out, dict):
                # 新版：clientKD 返回 {"mentor":(...), "mentee":(...)}
                add("mentor", out["mentor"])
                add("mentee", out["mentee"])
                if "mentor_posthoc" in out:
                    add("mentor_posthoc", out["mentor_posthoc"])
            else:
                # 旧版：只返回一份（默认视为 mentee/全局模型的度量）
                add("mentee", out)

            ids.append(c.id)

        # 把 mentor 的聚合先缓存到 self 里，evaluate() 里写入 TensorBoard
        self._mentor_agg = {
            "ns": mentor_ns,
            "top1": mentor_top1 / max(mentor_ns, 1.0),
            "top5": mentor_top5 / max(mentor_ns, 1.0),
            "auc": (mentor_auc_w / mentor_ns) if (self.num_classes >= 2 and mentor_ns > 0) else None,
        }
        self._mentor_posthoc_agg = {
            "ns": mentor_posthoc_ns,
            "top1": mentor_posthoc_top1 / max(mentor_posthoc_ns, 1.0),
            "top5": mentor_posthoc_top5 / max(mentor_posthoc_ns, 1.0),
            "auc": (mentor_posthoc_auc_w / mentor_posthoc_ns) if (self.num_classes >= 2 and mentor_posthoc_ns > 0) else None,
        }
        # 保持原有返回结构：只返回 mentee 的五元（id 列表 + 四个数组/权重）
        # 为了最小改动，这里仍旧按“列表”形式返回
        return (
            ids,
            [mentee_ns],                 # num_samples
            [mentee_top1],               # top1_counts（总数，evaluate 里会再 / sum(ns)）
            [mentee_top5],               # top5_counts
            [mentee_auc_w],              # auc_weighted
        )


    def evaluate(self, acc=None, loss=None):
        # mentee 主度量
        g1a = float("nan")
        g5a = float("nan")
        g1b = float("nan")
        g5b = float("nan")
        ids, ns_list, top1_list, top5_list, auc_w_list = self.test_metrics()
        stats_train = self.train_metrics()

        ns_total   = sum(ns_list)
        test_top1  = sum(top1_list) / max(ns_total, 1.0)
        test_top5  = sum(top5_list) / max(ns_total, 1.0)
        test_auc   = (sum(auc_w_list) / ns_total) if (self.num_classes >= 2 and ns_total > 0) else 0.0
        train_loss = sum(stats_train[2]) / max(sum(stats_train[1]), 1)

        # 兼容原来的记录：沿用 mentee 的 top1 作为“acc”
        self.rs_test_acc.append(test_top1)
        self.rs_train_loss.append(train_loss)
        if self.num_classes >= 2 and test_auc is not None:
            self.rs_test_auc.append(test_auc)

        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"[Mentee] Averaged Test Top1 : {test_top1:.4f}")
        print(f"[Mentee] Averaged Test Top5 : {test_top5:.4f}")
        if test_auc is not None:
            print(f"[Mentee] Averaged Test AUC  : {test_auc:.4f}")

        # 额外：把 mentor 也写进日志（若可得）
        r = getattr(self, "current_round", len(self.rs_test_acc)-1)
        self.tb.add_scalar("test/top1",      test_top1,  r)  # mentee 主曲线（保持原 tag）
        self.tb.add_scalar("test/top5",      test_top5,  r)
        if test_auc is not None:
            self.tb.add_scalar("test/avg_auc", test_auc,  r)
        self.tb.add_scalar("train/avg_loss", train_loss, r)

        if hasattr(self, "_mentor_agg") and self._mentor_agg["ns"] > 0:
            self.tb.add_scalar("test/mentor_top1", self._mentor_agg["top1"], r)
            self.tb.add_scalar("test/mentor_top5", self._mentor_agg["top5"], r)
            if self._mentor_agg["auc"] is not None:
                self.tb.add_scalar("test/mentor_auc", self._mentor_agg["auc"], r)
        if hasattr(self, "_mentor_posthoc_agg") and self._mentor_posthoc_agg.get("ns", 0) > 0:
            self.tb.add_scalar("test/mentor_posthoc_top1", self._mentor_posthoc_agg["top1"], r)
            self.tb.add_scalar("test/mentor_posthoc_top5", self._mentor_posthoc_agg["top5"], r)
            if hasattr(self, "_mentor_agg") and self._mentor_agg.get("ns", 0) > 0:
                self.tb.add_scalar("test/mentor_posthoc_delta", self._mentor_posthoc_agg["top1"] - self._mentor_agg["top1"], r)
        # -------- Global Test（全局测试集）额外评测：对齐论文口径 --------
        # g1, g5 = self._eval_global_test_with_model(self.global_model)
        # if g1 is not None:
        #     self.tb.add_scalar("test/global_top1", g1, r)
        #     self.tb.add_scalar("test/global_top5", g5, r)
        #     print(f"[GlobalTest] Top1={g1:.4f} Top5={g5:.4f}")
        g1a, g5a = self._eval_global_test_with_model(self.global_model, bn_recalib=False)
        if g1a is not None:
            self.tb.add_scalar("test/global_top1_noBN", g1a, r)
            self.tb.add_scalar("test/global_top5_noBN", g5a, r)
            print(f"[GlobalTest][noBN] Top1={g1a:.4f} Top5={g5a:.4f}")
        mentor_top1 = float('nan')
        mentor_top5 = float('nan')
        if hasattr(self, "_mentor_agg") and self._mentor_agg.get("ns", 0) > 0:
            mentor_top1 = float(self._mentor_agg.get("top1", float('nan')))
            mentor_top5 = float(self._mentor_agg.get("top5", float('nan')))
        g1b, g5b = self._eval_global_test_with_model(self.global_model, bn_recalib=True)
           # --- jitter metrics calculation ... ---
        # self._mentor_top1_hist.append(mentor_top1)
        # self._mentor_top5_hist.append(mentor_top5)
        # m_std, m_tv, m_tvmean = self._jitter_stats(self._mentor_top1_hist, self.jitter_window)
        # self.tb.add_scalar("jitter/mentor_top1_std", m_std, r)
        # self.tb.add_scalar("jitter/mentor_top1_TV", m_tv, r)
        # self.tb.add_scalar("jitter/mentor_top1_TVmean", m_tvmean, r)
        mentor_posthoc_top1 = float('nan')
        if hasattr(self, "_mentor_posthoc_agg") and self._mentor_posthoc_agg.get("ns", 0) > 0:
            mentor_posthoc_top1 = float(self._mentor_posthoc_agg.get("top1", float('nan')))
        mentor_posthoc_delta = mentor_posthoc_top1 - mentor_top1
        rec = {
            "eval_idx": int(r),
            "global_round": int(getattr(self, "current_round", r)),
            "mentee_local_top1": float(test_top1),
            "mentor_local_top1": mentor_top1,
            "mentor_posthoc_local_top1": mentor_posthoc_top1,
            "mentor_posthoc_delta": mentor_posthoc_delta,
            "global_noBN_top1": float(g1a) if g1a is not None else float('nan'),
            "global_BN_top1": float(g1b) if g1b is not None else float('nan'),
            "global_BN_top5": float(g5b) if g5b is not None else float('nan'),
            # "mentor_jitter_std": float(m_std),
            # "mentor_jitter_TV": float(m_tv),
            # "mentor_jitter_TVmean": float(m_tvmean),
        }
        self._append_eval_record(rec)
       
        if g1b is not None:
            self.tb.add_scalar("test/global_top1_BNcalib", g1b, r)
            self.tb.add_scalar("test/global_top5_BNcalib", g5b, r)
            print(f"[GlobalTest][BNcalib] Top1={g1b:.4f} Top5={g5b:.4f}")
    @staticmethod
    def _set_global_seed(seed: int, deterministic: bool = False):
        try:
            os.environ["PYTHONHASHSEED"] = str(int(seed))
        except Exception:
            pass
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    @staticmethod
    def _jitter_stats(seq, window: int):
        if window is None or int(window) <= 1:
            return float('nan'), float('nan'), float('nan')
        w = int(window)
        arr = np.asarray(list(seq)[-w:], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float('nan'), float('nan'), float('nan')
        if arr.size == 1:
            return 0.0, 0.0, 0.0
        std = float(np.std(arr))
        diffs = np.diff(arr)
        tv = float(np.sum(np.abs(diffs)))
        tv_mean = float(tv / max(arr.size - 1, 1))
        return std, tv, tv_mean
    def _append_eval_record(self, rec: dict):
        self.eval_records.append(rec)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(
                f"{rec.get('eval_idx',-1)}\t{rec.get('global_round',-1)}\t"
                f"{rec.get('mentee_local_top1',float('nan')):.6f}\t"
                f"{rec.get('mentor_local_top1',float('nan')):.6f}\t"
                f"{rec.get('mentor_posthoc_local_top1',float('nan')):.6f}\t"
                f"{rec.get('mentor_posthoc_delta',float('nan')):.6f}\t"
                f"{rec.get('global_noBN_top1',float('nan')):.6f}\t"
                f"{rec.get('global_BN_top1',float('nan')):.6f}\t"
                f"{rec.get('global_BN_top5',float('nan')):.6f}\n"
            )
    @staticmethod
    def update_bn_stats(model, loader, device):
        model.train() # 必须切换到 train 模式才能更新 running_mean/var
        print("Re-computing BN statistics...", end="")
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if type(x) == list: x = x[0] # 兼容你的 list 格式
                x = x.to(device)
                # 只需要做 forward，PyTorch 会自动更新 BN 统计量
                _ = model(x)
                # 跑个 50-100 个 batch 就足够收敛了，不需要跑完整个测试集
                if i > 50: 
                    break
        print("Done.")
        model.eval() # 记得切回 eval 模式
    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    def _model_bytes(self, model):
        sd = model.state_dict()
        total = 0
        for _, t in sd.items():
            if hasattr(t, "numel") and hasattr(t, "element_size"):
                total += int(t.numel() * t.element_size())
        return int(total)
