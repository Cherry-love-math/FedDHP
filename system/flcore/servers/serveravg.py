import time
import json
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import random
import os
class FedAvg(Server):
    def __init__(self, args, times):
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.svd_decomp_time = 0.0
        self.svd_recover_time = 0.0
    def send_models(self):
        assert len(self.selected_clients) > 0
        per_client = self._payload_nbytes(self.global_model)
        down = per_client * len(self.selected_clients)
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
        self._comm_down(down)
        self.total_client_rounds += int(len(self.selected_clients))
      #  self.total_selected_client_rounds += int(len(self.selected_clients))

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )
        per_client = self._payload_nbytes(self.global_model)
        up = per_client * len(active_clients)
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

        if tot_samples > 0:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples

        mbytes = int(self._payload_nbytes(self.global_model))
        up_success = int(mbytes) * int(len(self.uploaded_models))
        self._comm_up(up_success)

        sel = int(len(self.selected_clients))
        act_sampled = int(len(active_clients))
        act_uploaded = int(len(self.uploaded_models))
        down = int(self.round_downlink_bytes[-1]) if len(self.round_downlink_bytes) else 0
        up = up_success

        print(
            f"[CommDbg] round={self.current_round} sel={sel} "
            f"act_sampled={act_sampled} act_uploaded={act_uploaded} "
            f"down_total_MB={down/(1024*1024):.3f} up_total_MB={up/(1024*1024):.3f} "
            f"down_per_client_MB={down/(1024*1024)/max(sel,1):.3f} "
            f"up_per_uploaded_MB={up/(1024*1024)/max(act_uploaded,1):.3f}"
        )
     #   self.total_active_client_rounds += int(len(active_clients))


    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap  # 0,1,2,... 让 test 曲线横轴回到 0~24
                eval_t = time.time()
                self.evaluate()
                eval_cost += time.time() - eval_t
                self.current_round = _orig_round         # 立刻恢复，后面 comm 还是按真实轮次走

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
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
        
        total_up = int(sum(self.round_uplink_bytes))
        total_down = int(sum(self.round_downlink_bytes))
        avg_per_client_round = (total_up + total_down) / max(1, self.total_client_rounds)
        avg_per_client = (total_up + total_down) / max(1, self.num_clients)
        stats = {
            "total_uplink_bytes": total_up,
            "total_downlink_bytes": total_down,
            "total_bytes": total_up + total_down,
            "total_client_rounds": int(self.total_client_rounds),
            "avg_bytes_per_client_round": float(avg_per_client_round),
            "avg_bytes_per_client": float(avg_per_client),
        }

        save_dir = getattr(self, "save_path", None) or getattr(self, "save_folder_name", None) or "."
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "comm_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        if hasattr(self, "tb") and self.tb is not None:
            try:
                self.tb.flush()
            except Exception:
                pass

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
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
            "method": "FedAvg",
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

    def _payload_nbytes(self, obj):
        import numpy as np
        import torch
        if obj is None:
            return 0
        if isinstance(obj, (bytes, bytearray)):
            return int(len(obj))
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if torch.is_tensor(obj):
            return int(obj.numel() * obj.element_size())
        if isinstance(obj, (list, tuple)):
            s = 0
            for x in obj:
                s += self._payload_nbytes(x)
            return int(s)
        if isinstance(obj, dict):
            s = 0
            for v in obj.values():
                s += self._payload_nbytes(v)
            return int(s)
        if hasattr(obj, "state_dict") and callable(obj.state_dict):
            return self._payload_nbytes(obj.state_dict())
        return 0

    def _tb_add(self, tag, val, step):
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar(tag, val, step)

    def _comm_init_if_needed(self):
        if not hasattr(self, "round_downlink_bytes"):
            self.round_downlink_bytes = []
        if not hasattr(self, "round_uplink_bytes"):
            self.round_uplink_bytes = []

    def _comm_down(self, down_bytes):
        self._comm_init_if_needed()
        down_bytes = int(down_bytes)
        self.round_downlink_bytes.append(down_bytes)
        step = getattr(self, "current_round", len(self.round_downlink_bytes) - 1)
        self._tb_add("comm/downlink_MB", down_bytes / (1024 * 1024), step)

    def _comm_up(self, up_bytes):
        self._comm_init_if_needed()
        up_bytes = int(up_bytes)
        self.round_uplink_bytes.append(up_bytes)
        step = getattr(self, "current_round", len(self.round_uplink_bytes) - 1)
        self._tb_add("comm/uplink_MB", up_bytes / (1024 * 1024), step)
        down = self.round_downlink_bytes[-1] if len(self.round_downlink_bytes) else 0
        self._tb_add("comm/total_MB", (down + up_bytes) / (1024 * 1024), step)
