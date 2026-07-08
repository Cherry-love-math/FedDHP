import time
import numpy as np
import random
import torch
import cvxpy as cvx
import copy
from flcore.clients.clientpac import clientPAC
from flcore.servers.serverbase import Server
from threading import Thread
from collections import defaultdict
import os, json

class FedPAC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPAC)
        self._round_up_bytes = 0
        self._round_down_bytes = 0
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.svd_decomp_time = 0.0
        self.svd_recover_time = 0.0
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]

        self.Vars = []
        self.Hs = []
        self.uploaded_heads = []


    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self._round_up_bytes = 0
            self._round_down_bytes = 0
            self.selected_clients = self.select_clients()
            self.send_models()

            self.Vars = []
            self.Hs = []
            for client in self.selected_clients:
                self.Vars.append(client.V)
                self.Hs.append(client.h)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
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

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.receive_models()
            self.aggregate_parameters()

            self.aggregate_and_send_heads()
            self._commit_round_comm()
            round_wall = time.time() - s_t
            train_wall = round_wall - eval_cost
            self.Budget.append(round_wall)
            self.Budget_train.append(train_wall)
            self.Budget_eval.append(eval_cost)
            print('-'*50, self.Budget[-1])
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
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
       
        self.save_results()
        total_up = int(sum(self.round_uplink_bytes)) if hasattr(self, "round_uplink_bytes") else 0
        total_down = int(sum(self.round_downlink_bytes)) if hasattr(self, "round_downlink_bytes") else 0

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
            "method": "FedPAC",
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

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        new_base = copy.deepcopy(self.global_model.base)
        for p in new_base.parameters():
            p.data.zero_()

        for w, client_base in zip(self.uploaded_weights, self.uploaded_models):
            for sp, cp in zip(new_base.parameters(), client_base.parameters()):
                sp.data += cp.data.clone() * w

        self.global_model.base = new_base
    # 2. 聚合 Head (仅用于组装一个完整的 Global Model 给 global_test 评估用)
        new_head = copy.deepcopy(self.global_model.head)
        for p in new_head.parameters():
            p.data.zero_()
        
        for w, client_head in zip(self.uploaded_weights, self.uploaded_heads):
            for sp, cp in zip(new_head.parameters(), client_head.parameters()):
                sp.data += cp.data.clone() * w
                
        self.global_model.head = new_head
    def _commit_round_comm(self):
        self._comm_init_if_needed()

        down = int(getattr(self, "_round_down_bytes", 0))
        up = int(getattr(self, "_round_up_bytes", 0))

        self.round_downlink_bytes.append(down)
        self.round_uplink_bytes.append(up)

        r = getattr(self, "current_round", len(self.round_downlink_bytes) - 1)

        self._tb_add("comm/downlink_MB", down / (1024 * 1024), r)
        self._tb_add("comm/uplink_MB",   up   / (1024 * 1024), r)
        self._tb_add("comm/total_MB",   (down + up) / (1024 * 1024), r)

    def send_models(self):
        assert len(self.selected_clients) > 0

        per_client = self._payload_nbytes(self.global_model.base if hasattr(self.global_model, "base") else self.global_model)
        down = per_client * len(self.selected_clients)

        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model.base if hasattr(self.global_model, "base") else self.global_model)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

        self._round_down_bytes += int(down)
        self.total_client_rounds += int(len(self.selected_clients))

    def send_protos(self):
        assert len(self.clients) > 0

        per_client = self._payload_nbytes(self.global_protos)
        down = per_client * len(self.clients)

        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

        self._round_down_bytes += int(down)
   #     self.total_client_rounds += int(len(self.clients))

    def receive_protos(self):
        assert len(self.selected_clients) > 0

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
        up = 0
        for p in self.uploaded_protos:
            up += self._payload_nbytes(p)
        self._round_up_bytes += int(up)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_heads = []
        tot_samples = 0

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"] + \
                                   client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
                self.uploaded_heads.append(client.model.head)

        if tot_samples > 0:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples

        if len(self.uploaded_models) > 0:
            per_base = self._payload_nbytes(self.uploaded_models[0])
            per_head = self._payload_nbytes(self.uploaded_heads[0])
            up = (per_base + per_head) * len(self.uploaded_models)
        else:
            up = 0

        self._round_up_bytes += int(up)

        sel = int(len(self.selected_clients))
        act_sampled = int(len(active_clients))
        act_uploaded = int(len(self.uploaded_models))
    #    down = int(self.round_downlink_bytes[-1]) if len(self.round_downlink_bytes) else 0
        down = int(getattr(self, "_round_down_bytes", 0))
        print(
            f"[CommDbg] round={getattr(self,'current_round',-1)} sel={sel} "
            f"act_sampled={act_sampled} act_uploaded={act_uploaded} "
            f"down_total_MB={down/(1024*1024):.3f} up_total_MB={up/(1024*1024):.3f} "
            f"down_per_client_MB={down/(1024*1024)/max(sel,1):.3f} "
            f"up_per_uploaded_MB={up/(1024*1024)/max(act_uploaded,1):.3f}"
        )

#     def evaluate(self, acc=None, loss=None):
#         stats = self.test_metrics()
#         # stats_train = self.train_metrics()

#         test_acc = sum(stats[2])*1.0 / sum(stats[1])
#         # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
#         accs = [a / n for a, n in zip(stats[2], stats[1])]
        
#         if acc == None:
#             self.rs_test_acc.append(test_acc)
#         else:
#             acc.append(test_acc)
        
#         # if loss == None:
#         #     self.rs_train_loss.append(train_loss)
#         # else:
#         #     loss.append(train_loss)

#         # print("Averaged Train Loss: {:.4f}".format(train_loss))
#         print("Averaged Test Accuracy: {:.4f}".format(test_acc))
#         # self.print_(test_acc, train_acc, train_loss)
#         print("Std Test Accuracy: {:.4f}".format(np.std(accs)))

    def aggregate_and_send_heads(self):
        head_weights = solve_quadratic(len(self.uploaded_ids), self.Vars, self.Hs)
        down = 0 
        for idx, cid in enumerate(self.uploaded_ids):
            if self.current_round % 5 == 0:
                print('(Client {}) Weights of Classifier Head'.format(cid))
                print(head_weights[idx],'\n')

            if head_weights[idx] is not None:
                new_head = self.add_heads(head_weights[idx])
            else:
                new_head = self.uploaded_heads[idx]
            down += self._payload_nbytes(new_head)
            self.clients[cid].set_head(new_head)
        self._round_down_bytes += int(down)
       # self.total_client_rounds += int(len(self.uploaded_ids))
    def add_heads(self, weights):
        new_head = copy.deepcopy(self.uploaded_heads[0])
        for param in new_head.parameters():
            param.data.zero_()
                    
        for w, head in zip(weights, self.uploaded_heads):
            for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                server_param.data += client_param.data.clone() * w
        return new_head

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

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L94
def solve_quadratic(num_users, Vars, Hs):
    device = Hs[0].device
    v = torch.tensor(Vars, device=device, dtype=Hs[0].dtype)
    Vdiag = torch.diag(v)
    avg_weight = []

    for i in range(num_users):
        h_ref = Hs[i]
        delta = torch.stack([(h_ref - Hs[j]).reshape(-1) for j in range(num_users)], dim=0)
        dist = delta @ delta.t()

        p_matrix = (Vdiag + dist).detach().cpu().numpy()
        p_matrix = 0.5 * (p_matrix + p_matrix.T)

        evals, evecs = np.linalg.eigh(p_matrix)
        if np.min(evals) < -1e-8:
            evals = np.maximum(evals, 1e-2)
            p_matrix = (evecs * evals) @ evecs.T
            p_matrix = 0.5 * (p_matrix + p_matrix.T)

        alpha = None
        eps = 1e-3

        try:
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve(solver=cvx.OSQP, warm_start=True, verbose=False)
            if alphav.value is not None:
                alpha = [float(a) if a > eps else 0.0 for a in alphav.value]
        except Exception:
            alpha = None

        avg_weight.append(alpha)

    return avg_weight
# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L10
def pairwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])
