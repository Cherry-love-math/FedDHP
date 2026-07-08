import time
import random
import json
import numpy as np
from flcore.clients.clientrod import clientROD
from flcore.servers.serverbase import Server
from threading import Thread


class FedROD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientROD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.svd_decomp_time = 0.0
        self.svd_recover_time = 0.0

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        up = 0
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
                try:
                    up += int(self._model_bytes(client.model))
                except Exception:
                    pass
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        # --- [最小改动] 强制改为均匀加权 ---
        # 此时 self.uploaded_weights 的长度就是有效客户端的数量 N
        # 我们忽略之前收集的样本数 w，直接赋值为 1.0 / N
        n_models = len(self.uploaded_weights)
        if n_models > 0:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = 1.0 / n_models
        # ---------------------------------
        self.round_uplink_bytes.append(int(up))
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            down = self.round_downlink_bytes[-1] if self.round_downlink_bytes else 0
            self.tb.add_scalar(
                "comm/total_MB",
                (up + down) / (1024 * 1024),
                self.current_round,
            )
    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self.selected_clients = self.select_clients()

            # communication: downlink (full global model to selected clients)
            try:
                per_client = self._model_bytes(self.global_model)
            except Exception:
                per_client = 0
            down = int(per_client * len(self.selected_clients))
            self.round_downlink_bytes.append(down)
            self.total_client_rounds += int(len(self.selected_clients))
            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), i)

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap 
                eval_t = time.time()
                self.evaluate()
                eval_cost += time.time() - eval_t
                self.current_round = _orig_round
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

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
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage raw time cost per round.")
        print(float(np.mean(self.Budget[1:])) if len(self.Budget) > 1 else float("nan"))

        print("\nAverage training time per round excluding evaluation.")
        print(float(np.mean(self.Budget_train[1:])) if len(self.Budget_train) > 1 else float("nan"))


        self.save_results()
        self.save_comm_stats()
        
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientROD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
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
            "method": "FedROD",
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
        print(f"[Comm] Down total: {total_down / (1024*1024):.3f} MB")
        print(f"[Comm] Up   total: {total_up / (1024*1024):.3f} MB")
        print(f"[Comm] Total     : {total_bytes / (1024*1024):.3f} MB")
        print(f"[Time] Train excl. eval: {stats['total_train_wall_time_excl_eval_sec']:.3f} s")
        print(f"[Time] Avg client train: {stats['avg_client_train_time_sec']:.3f} s")