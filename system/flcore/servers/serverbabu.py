import copy
import time
import random
import json
from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
from threading import Thread
import torch


class FedBABU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientBABU)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.final_head_ft_time = 0.0
    def send_models(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model.base)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
    def aggregate_parameters(self):
        if len(self.uploaded_models) == 0:
            return
        base_accum = {}
        for k, v in self.global_model.base.state_dict().items():
            if torch.is_floating_point(v):
                base_accum[k] = v.detach().clone().zero_()
            else:
                base_accum[k] = v.detach().clone()
        for w, client_base in zip(self.uploaded_weights, self.uploaded_models):
            sd = client_base.state_dict()
            ww = float(w)
            for k in base_accum.keys():
                if not torch.is_floating_point(base_accum[k]):
                    continue
                base_accum[k].add_(sd[k].to(dtype=base_accum[k].dtype), alpha=ww)
        self.global_model.base.load_state_dict(base_accum, strict=True)

    def evaluate_babu_local(self):
        total_top1 = 0.0
        total_top5 = 0.0
        total_n = 0.0
        for client in self.clients:
            out = client.test_metrics()
            if len(out) == 3:
                top1, n, auc = out
                top5 = top1
            else:
                top1, n, auc, top5 = out
            total_top1 += float(top1)
            total_top5 += float(top5)
            total_n += float(n)
        return {
            "top1": total_top1 / max(total_n, 1.0),
            "top5": total_top5 / max(total_n, 1.0),
            "n": total_n,
        }

    def _restore_client_states(self, saved_model_states, saved_optimizer_states=None):
        for idx, client in enumerate(self.clients):
            client.model.load_state_dict(copy.deepcopy(saved_model_states[idx]), strict=True)
            if saved_optimizer_states is not None and hasattr(client, "optimizer"):
                client.optimizer.load_state_dict(copy.deepcopy(saved_optimizer_states[idx]))

    def _run_babu_ft_variant(self, tag, saved_model_states, saved_optimizer_states=None, sync_global_base=True, which_module=['head'], optimizer_mode='fresh', ft_lr=None):
        self._restore_client_states(saved_model_states, saved_optimizer_states)
        if sync_global_base:
            for client in self.clients:
                client.set_parameters(self.global_model.base)
        preft = self.evaluate_babu_local()
        ft_t = time.time()
        for client in self.clients:
            client.fine_tune(which_module=which_module, optimizer_mode=optimizer_mode, ft_lr=ft_lr)
        ft_time = time.time() - ft_t
        postft = self.evaluate_babu_local()
        delta = postft["top1"] - preft["top1"]
        print(f"[FedBABU-Diag][{tag}] pre={preft['top1']:.4f}, post={postft['top1']:.4f}, delta={delta:.4f}, ft_time={ft_time:.3f}s")
        return {
            "tag": tag,
            "sync_global_base": bool(sync_global_base),
            "which_module": list(which_module),
            "optimizer_mode": optimizer_mode,
            "ft_lr": None if ft_lr is None else float(ft_lr),
            "pre_top1": float(preft["top1"]),
            "post_top1": float(postft["top1"]),
            "delta_top1": float(delta),
            "ft_time_sec": float(ft_time),
        }

    def _write_babu_diag_tsv(self, diag_results):
        diag_path = self.logdir / "fedbabu_ft_diagnostic.tsv"
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write("tag\tsync_global_base\twhich_module\toptimizer_mode\tft_lr\tpre_top1\tpost_top1\tdelta_top1\tft_time_sec\n")
            for r in diag_results:
                which_module = "+".join(r.get("which_module", []))
                ft_lr = r.get("ft_lr", None)
                ft_lr_str = "" if ft_lr is None else f"{float(ft_lr):.8g}"
                f.write(
                    f"{r.get('tag','')}\t"
                    f"{int(bool(r.get('sync_global_base', False)))}\t"
                    f"{which_module}\t"
                    f"{r.get('optimizer_mode','')}\t"
                    f"{ft_lr_str}\t"
                    f"{float(r.get('pre_top1', float('nan'))):.6f}\t"
                    f"{float(r.get('post_top1', float('nan'))):.6f}\t"
                    f"{float(r.get('delta_top1', float('nan'))):.6f}\t"
                    f"{float(r.get('ft_time_sec', float('nan'))):.6f}\n"
                )
        return diag_path

    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self.selected_clients = self.select_clients()

            # communication: downlink (global base to selected clients)
            try:
                per_client = self._model_bytes(self.global_model.base)
            except Exception:
                per_client = 0
            down = int(per_client * len(self.selected_clients))
            self.round_downlink_bytes.append(down)
            self.total_client_rounds += int(len(self.selected_clients))
            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap
                eval_t = time.time()
                self.evaluate()
                eval_cost += time.time() - eval_t
                if len(self.rs_test_acc) > 0 and hasattr(self, "tb") and self.tb is not None:
                    self.tb.add_scalar("test/mentor_top1", self.rs_test_acc[-1], self.current_round)
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
            train_wall = max(0.0, round_wall - eval_cost)
            self.Budget.append(round_wall)
            self.Budget_train.append(train_wall)
            self.Budget_eval.append(eval_cost)
            print('-'*25, 'time cost', '-'*25, f"raw={round_wall:.3f}s, train_excl_eval={train_wall:.3f}s, eval={eval_cost:.3f}s")
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

        # for client in self.clients:
        #     client.fine_tune()
        # print("\n-------------Evaluate fine-tuned personalized models-------------")
        # self.current_round = (self.global_rounds // self.eval_gap) + 1
        # self.evaluate()
        saved_model_states = [copy.deepcopy(client.model.state_dict()) for client in self.clients]
        saved_optimizer_states = [copy.deepcopy(client.optimizer.state_dict()) for client in self.clients]

        diag_results = []
        diag_results.append(self._run_babu_ft_variant(
            "A_global_head_fresh",
            saved_model_states,
            saved_optimizer_states,
            sync_global_base=True,
            which_module=['head'],
            optimizer_mode='fresh',
        ))
        diag_results.append(self._run_babu_ft_variant(
            "B_global_base_head_fresh",
            saved_model_states,
            saved_optimizer_states,
            sync_global_base=True,
            which_module=['base', 'head'],
            optimizer_mode='fresh',
        ))
        diag_results.append(self._run_babu_ft_variant(
            "C_global_head_oldopt",
            saved_model_states,
            saved_optimizer_states,
            sync_global_base=True,
            which_module=['head'],
            optimizer_mode='old',
        ))
        diag_results.append(self._run_babu_ft_variant(
            "D_local_base_head_oldopt",
            saved_model_states,
            saved_optimizer_states,
            sync_global_base=False,
            which_module=['base', 'head'],
            optimizer_mode='old',
        ))

        official = diag_results[0]
        preft = {"top1": official["pre_top1"]}
        postft = {"top1": official["post_top1"]}
        delta = official["delta_top1"]
        self.final_head_ft_time = official["ft_time_sec"]

        print("\n-------------Evaluate FedBABU head-adapted personalized models-------------")
        print(f"[FedBABU] mentor_top1(preFT)={preft['top1']:.4f}, mentor_posthoc_top1(postFT)={postft['top1']:.4f}, mentor_posthoc_delta={delta:.4f}")

        step = (self.global_rounds // self.eval_gap) + 1
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("test/mentor_top1", preft["top1"], step)
            self.tb.add_scalar("test/mentor_posthoc_top1", postft["top1"], step)
            self.tb.add_scalar("test/mentor_posthoc_delta", delta, step)
            self.tb.add_scalar("time/final_head_ft_sec", self.final_head_ft_time, step)

        summary = {
            "method": "FedBABU",
            "mentor_top1": float(preft["top1"]),
            "mentor_posthoc_top1": float(postft["top1"]),
            "mentor_posthoc_delta": float(delta),
            "fine_tuning_epochs": int(getattr(self.args, "fine_tuning_epochs", 0)),
            "adapted_part": "head",
            "protocol": "FedBABU native head adaptation from final global base",
            "final_head_ft_time_sec": float(self.final_head_ft_time)
        }
        save_path = self.logdir / "posthoc_summary.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        diag_path = self._write_babu_diag_tsv(diag_results)
        print(f"[FedBABU] summary saved to {save_path}")
        print(f"[FedBABU] diagnostic TSV saved to {diag_path}")

        self._restore_client_states(saved_model_states, saved_optimizer_states)
        for client in self.clients:
            client.set_parameters(self.global_model.base)
        self.save_comm_stats()
        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientBABU)
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
                self.uploaded_models.append(client.model.base)
                try:
                    up += int(self._model_bytes(client.model.base))
                except Exception:
                    pass
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        self.round_uplink_bytes.append(int(up))
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            self.tb.add_scalar(
                "comm/total_MB",
                (up + (self.round_downlink_bytes[-1] if self.round_downlink_bytes else 0)) / (1024 * 1024),
                self.current_round,
            )
    def save_comm_stats(self):
        total_up = int(sum(self.round_uplink_bytes))
        total_down = int(sum(self.round_downlink_bytes))
        total_bytes = total_up + total_down
        budget_raw = getattr(self, "Budget", [])
        budget_train = getattr(self, "Budget_train", [])
        budget_eval = getattr(self, "Budget_eval", [])
        total_train = float(sum(budget_train))
        final_ft = float(getattr(self, "final_head_ft_time", 0.0))
        stats = {
            "total_uplink_bytes": total_up,
            "total_downlink_bytes": total_down,
            "total_bytes": total_bytes,
            "total_client_rounds": int(getattr(self, "total_client_rounds", 0)),
            "avg_bytes_per_client_round": float(total_bytes / max(1, getattr(self, "total_client_rounds", 1))),
            "avg_bytes_per_client": float(total_bytes / max(1, self.num_clients)),
            "total_round_wall_time_sec": float(sum(budget_raw)),
            "avg_round_wall_time_sec": float(sum(budget_raw[1:]) / max(len(budget_raw) - 1, 1)) if len(budget_raw) > 1 else float("nan"),
            "total_train_wall_time_excl_eval_sec": total_train,
            "avg_train_wall_time_excl_eval_sec": float(sum(budget_train[1:]) / max(len(budget_train) - 1, 1)) if len(budget_train) > 1 else float("nan"),
            "total_eval_wall_time_sec": float(sum(budget_eval)),
            "final_head_ft_time_sec": final_ft,
            "end_to_end_personalized_time_sec": total_train + final_ft
        }
        save_path = self.logdir / "comm_stats.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[Stats] FedBABU efficiency stats saved to {save_path}")
        print(f"[Time] Train excl. eval: {stats['total_train_wall_time_excl_eval_sec']:.3f} s")
        print(f"[Time] Final head FT: {stats['final_head_ft_time_sec']:.3f} s")
        print(f"[Time] End-to-end personalized: {stats['end_to_end_personalized_time_sec']:.3f} s")