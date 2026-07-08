import copy
import json
import numpy as np
import torch
import torch.nn as nn
import time
from flcore.clients.clientgpfl import *
from flcore.servers.serverbase import Server
from threading import Thread
import torch.nn.functional as F
class GPFLServerGenericEval(nn.Module):
    def __init__(self, base, cov, gce):
        super().__init__()
        self.base = base
        self.cov = cov
        self.gce = gce

    def forward(self, x):
        feat = self.base(x)
        emb = self.gce.embedding(torch.arange(self.gce.num_classes, device=feat.device))
        generic_input = emb.mean(dim=0, keepdim=True).expand(feat.size(0), -1)
        feat_g = self.cov(feat, generic_input)
        logits = F.linear(F.normalize(feat_g, dim=1), F.normalize(emb, dim=1))
        return logits
class GPFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]

        args.GCE = GCE(
            in_features=self.feature_dim,
            num_classes=args.num_classes,
            dev=args.device
        ).to(args.device)

        args.CoV = CoV(self.feature_dim).to(args.device)

        self.GCE = copy.deepcopy(args.GCE)
        self.CoV = copy.deepcopy(args.CoV)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGPFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0
        self.svd_decomp_time = 0.0
        self.svd_recover_time = 0.0


    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self.selected_clients = self.select_clients()

            # if i%self.eval_gap == 0:
            #     print(f"\n-------------Round number: {i}-------------")
            #     print("\nEvaluate performance")
            #     eval_t = time.time()
            #     self.evaluate()
            #     eval_cost += time.time() - eval_t
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate performance")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap
                eval_t = time.time()
                self.evaluate()
                eval_cost += time.time() - eval_t
                self.current_round = _orig_round
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            up = self._calc_selected_payload_bytes()
            down = self._calc_downlink_payload_bytes()
            self.round_uplink_bytes.append(int(up))
            self.round_downlink_bytes.append(int(down))
            self.total_client_rounds += int(len(self.selected_clients))

            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
                self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)
                self.tb.add_scalar("comm/total_MB", (up + down) / (1024 * 1024), self.current_round)

            self.send_models()

            self.global_GCE()
            self.global_CoV()

            round_wall = time.time() - s_t
            train_wall = round_wall - eval_cost
            self.Budget.append(round_wall)
            self.Budget_train.append(train_wall)
            self.Budget_eval.append(eval_cost)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("time/round_wall_sec", round_wall, self.current_round)
                self.tb.add_scalar("time/train_wall_excl_eval_sec", train_wall, self.current_round)
                self.tb.add_scalar("time/eval_wall_sec", eval_cost, self.current_round)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage raw time cost per round.")
        print(float(np.mean(self.Budget[1:])) if len(self.Budget) > 1 else float("nan"))
        print("\nAverage training time per round excluding evaluation.")
        print(float(np.mean(self.Budget_train[1:])) if len(self.Budget_train) > 1 else float("nan"))

        self.save_results()
        self.save_global_model()
        self.save_comm_stats()


    def _calc_selected_payload_bytes(self):
        total = 0
        for client in self.selected_clients:
            try:
                total += int(self._model_bytes(client.model.base))
                total += int(self._model_bytes(client.GCE))
                total += int(self._model_bytes(client.CoV))
            except Exception:
                pass
        return int(total)

    def _calc_downlink_payload_bytes(self):
        try:
            per_client = int(self._model_bytes(self.global_model))
        except Exception:
            per_client = 0
        try:
            per_client += int(self._model_bytes(self.GCE))
            per_client += int(self._model_bytes(self.CoV))
        except Exception:
            pass
        return int(per_client * len(self.selected_clients))

    def save_comm_stats(self):
        total_up = int(sum(getattr(self, "round_uplink_bytes", [])))
        total_down = int(sum(getattr(self, "round_downlink_bytes", [])))
        total_bytes = total_up + total_down
        budget_raw = getattr(self, "Budget", [])
        budget_train = getattr(self, "Budget_train", [])
        budget_eval = getattr(self, "Budget_eval", [])

        client_train_times = []
        client_send_times = []
        total_client_train = 0.0
        total_client_send = 0.0
        for client in self.clients:
            tr = getattr(client, "train_time_cost", {})
            sr = getattr(client, "send_time_cost", {})
            tr_rounds = int(tr.get("num_rounds", 0))
            sr_rounds = int(sr.get("num_rounds", 0))
            tr_total = float(tr.get("total_cost", 0.0))
            sr_total = float(sr.get("total_cost", 0.0))
            total_client_train += tr_total
            total_client_send += sr_total
            if tr_rounds > 0:
                client_train_times.append(tr_total / tr_rounds)
            if sr_rounds > 0:
                client_send_times.append(sr_total / sr_rounds)

        stats = {
            "method": "GPFL",
            "total_uplink_bytes": total_up,
            "total_downlink_bytes": total_down,
            "total_bytes": total_bytes,
            "total_client_rounds": int(getattr(self, "total_client_rounds", 0)),
            "avg_bytes_per_client_round": float(total_bytes / max(1, int(getattr(self, "total_client_rounds", 0)))),
            "total_round_wall_time_sec": float(sum(budget_raw)),
            "avg_round_wall_time_sec": float(np.mean(budget_raw[1:])) if len(budget_raw) > 1 else float("nan"),
            "total_train_wall_time_excl_eval_sec": float(sum(budget_train)),
            "avg_train_wall_time_excl_eval_sec": float(np.mean(budget_train[1:])) if len(budget_train) > 1 else float("nan"),
            "total_eval_wall_time_sec": float(sum(budget_eval)),
            "avg_client_train_time_sec": float(np.mean(client_train_times)) if len(client_train_times) else float("nan"),
            "total_client_train_time_sec": float(total_client_train),
            "avg_client_send_time_sec": float(np.mean(client_send_times)) if len(client_send_times) else float("nan"),
            "total_client_send_time_sec": float(total_client_send),
            "total_svd_decomp_time_sec": float(getattr(self, "svd_decomp_time", 0.0)),
            "total_svd_recover_time_sec": float(getattr(self, "svd_recover_time", 0.0)),
        }

        save_path = self.logdir / "comm_stats.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"[Stats] Efficiency stats saved to {save_path}")
        print(f"[Time] Train excl. eval: {stats['total_train_wall_time_excl_eval_sec']:.3f} s")
        print(f"[Time] Avg client train: {stats['avg_client_train_time_sec']:.3f} s")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.base)
            
    def global_GCE(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.GCE)

        self.GCE = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.GCE.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_GCE(w, client_model)

        for client in self.clients:
            client.set_GCE(self.GCE)

    def add_GCE(self, w, GCE):
        for server_param, client_param in zip(self.GCE.parameters(), GCE.parameters()):
            server_param.data += client_param.data.clone() * w
            
    def global_CoV(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.CoV)

        self.CoV = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.CoV.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_CoV(w, client_model)

        for client in self.clients:
            client.set_CoV(self.CoV)

    def add_CoV(self, w, CoV):
        for server_param, client_param in zip(self.CoV.parameters(), CoV.parameters()):
            server_param.data += client_param.data.clone() * w
    def _eval_global_test_with_model(self, model, bn_recalib=False):
        base = model.base if hasattr(model, "base") else model
        eval_model = GPFLServerGenericEval(base, self.CoV, self.GCE).to(self.device)
        return super()._eval_global_test_with_model(eval_model, bn_recalib=bn_recalib)

class GCE(nn.Module):
    def __init__(self, in_features, num_classes, dev='cpu'):
        super(GCE, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        self.dev = dev

    def forward(self, x, label):
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.dev))
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size(), device=self.dev)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = - torch.mean(torch.sum(softmax_loss, dim=1))

        return softmax_loss


class CoV(nn.Module):
    def __init__(self, in_dim):
        super(CoV, self).__init__()

        self.Conditional_gamma = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.Conditional_beta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.act = nn.ReLU()

    def forward(self, x, context):
        gamma = self.Conditional_gamma(context)
        beta = self.Conditional_beta(context)

        out = torch.multiply(x, gamma + 1)
        out = torch.add(out, beta)
        out = self.act(out)
        return out