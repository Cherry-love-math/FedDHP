import os
import time
import copy
import h5py
import random
import json
import numpy as np
import torch
from flcore.clients.clientpFedMe import clientpFedMe
from flcore.servers.serverbase import Server
from threading import Thread
class pFedMe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0
        self.set_slow_clients()
        self.set_clients(clientpFedMe)
        self.beta = args.beta
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_test_acc_per = []
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.svd_decomp_time = 0.0
        self.svd_recover_time = 0.0
    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        active_clients = random.sample(self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        up = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"] + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                try:
                    up += int(self._model_bytes(client.model))
                except Exception:
                    pass
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        self.round_uplink_bytes.append(int(up))
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            down = self.round_downlink_bytes[-1] if self.round_downlink_bytes else 0
            self.tb.add_scalar("comm/total_MB", (up + down) / (1024 * 1024), self.current_round)
    def train(self):
        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self.selected_clients = self.select_clients()
            try:
                per_client = self._model_bytes(self.global_model)
            except Exception:
                per_client = 0
            down = int(per_client * len(self.selected_clients))
            self.round_downlink_bytes.append(down)
            self.total_client_rounds += int(len(self.selected_clients))
            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)
            self.send_models()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap
                print("\nEvaluate global model")
                eval_t = time.time()
                self.evaluate()
                print("\nEvaluate personalized model")
                self.evaluate_personalized()
                eval_cost += time.time() - eval_t
                self.current_round = _orig_round
            for client in self.selected_clients:
                client.train()
            self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.beta_aggregate_parameters()
            round_wall = time.time() - s_t
            train_wall = round_wall - eval_cost
            self.Budget.append(round_wall)
            self.Budget_train.append(train_wall)
            self.Budget_eval.append(eval_cost)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])
            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("time/round_wall_sec", round_wall, self.current_round)
                self.tb.add_scalar("time/train_wall_excl_eval_sec", train_wall, self.current_round)
                self.tb.add_scalar("time/eval_wall_sec", eval_cost, self.current_round)
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc_per], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        print(max(self.rs_test_acc_per) if len(self.rs_test_acc_per) else 0.0)
        print("\nAverage raw time cost per round.")
        print(float(np.mean(self.Budget[1:])) if len(self.Budget) > 1 else float("nan"))
        print("\nAverage training time per round excluding evaluation.")
        print(float(np.mean(self.Budget_train[1:])) if len(self.Budget_train) > 1 else float("nan"))
        self.save_results()
        self.save_global_model()
        self.save_comm_stats()
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientpFedMe)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        if hasattr(self, "tb") and self.tb is not None:
            try:
                self.tb.flush()
            except Exception:
                pass
            self.tb.close()
            self.tb = None

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
            "method": "pFedMe",
            "total_uplink_bytes": total_up,
            "total_downlink_bytes": total_down,
            "total_bytes": total_bytes,
            "total_client_rounds": int(getattr(self, "total_client_rounds", 0)),
            "avg_bytes_per_client_round": float(total_bytes / max(1, int(getattr(self, "total_client_rounds", 0)))),
            "avg_bytes_per_client": float(total_bytes / max(1, int(getattr(self, "num_clients", 1)))),
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
        print(f"[Comm] Total: {total_bytes / (1024 * 1024):.3f} MB")
        print(f"[Time] Train excl. eval: {stats['total_train_wall_time_excl_eval_sec']:.3f} s")
        print(f"[Time] Avg client train: {stats['avg_client_train_time_sec']:.3f} s")

    def _backup_client_model(self, c):
        return [p.data.clone() for p in c.model.parameters()]
    def _restore_client_model(self, c, backup):
        with torch.no_grad():
            for p, b in zip(c.model.parameters(), backup):
                p.data.copy_(b)
    def test_metrics_global(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return super().test_metrics()
        ids = []
        num_samples = []
        tot_correct = []
        tot_top5 = []
        auc_w_list = []
        for c in self.clients:
            bk = self._backup_client_model(c)
            c.update_parameters(c.model, self.global_model.parameters())
            top1_correct, test_num, auc, top5_correct = c.test_metrics()
            self._restore_client_model(c, bk)
            ids.append(c.id)
            num_samples.append(test_num)
            tot_correct.append(float(top1_correct))
            tot_top5.append(float(top5_correct))
            auc_w_list.append((float(auc) if auc is not None else 0.0) * float(test_num))
        return ids, num_samples, tot_correct, tot_top5, auc_w_list
    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        ids = []
        num_samples = []
        tot_correct = []
        tot_top5 = []
        auc_w_list = []
        for c in self.clients:
            bk = self._backup_client_model(c)
            try:
                c.update_parameters(c.model, c.personalized_params)
            except Exception:
                pass
            top1_correct, test_num, auc, top5_correct = c.test_metrics_ft()
            self._restore_client_model(c, bk)
            ids.append(c.id)
            num_samples.append(test_num)
            tot_correct.append(float(top1_correct))
            tot_top5.append(float(top5_correct))
            auc_w_list.append((float(auc) if auc is not None else 0.0) * float(test_num))
        return ids, num_samples, tot_correct, tot_top5, auc_w_list
    def test_metrics(self):
        return self.test_metrics_global()
    def beta_aggregate_parameters(self):
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta) * pre_param.data + self.beta * param.data
    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0], [0]
        ids = []
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            bk = self._backup_client_model(c)
            try:
                c.update_parameters(c.model, c.personalized_params)
            except Exception:
                pass
            ct, cl, ns = c.train_metrics_personalized()
            ids.append(c.id)
            num_samples.append(ns)
            tot_correct.append(ct * 1.0)
            losses.append(cl * 1.0)
            self._restore_client_model(c, bk)
        return ids, num_samples, tot_correct, losses
    def evaluate(self, acc=None, loss=None):
        r = getattr(self, "current_round", 0)
        g1a, g5a = self._eval_global_test_with_model(self.global_model, bn_recalib=False)
        if g1a is not None and hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("test/global_top1_noBN", g1a, r)
            self.tb.add_scalar("test/global_top5_noBN", g5a, r)
        g1b, g5b = self._eval_global_test_with_model(self.global_model, bn_recalib=True)
        if g1b is not None and hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("test/global_top1_BNcalib", g1b, r)
            self.tb.add_scalar("test/global_top5_BNcalib", g5b, r)
        if g1b is not None:
            print(f"[GlobalTest][BNcalib] Top1={g1b:.4f} Top5={g5b:.4f}")
    def evaluate_personalized(self):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()
        ns_total = sum(stats[1])
        test_acc = sum(stats[2]) / max(ns_total, 1.0)
        train_acc = sum(stats_train[2]) / max(sum(stats_train[1]), 1.0)
        train_loss = sum(stats_train[3]) / max(sum(stats_train[1]), 1.0)
        self.rs_test_acc_per.append(test_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        r = getattr(self, "current_round", len(self.rs_test_acc_per) - 1)
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("test/top1", test_acc, r)
        self.print_(test_acc, train_acc, train_loss)
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if len(self.rs_test_acc_per):
            algo2 = algo + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo2), "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc_per)
                hf.create_dataset("rs_train_acc", data=self.rs_train_acc_per)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss_per)