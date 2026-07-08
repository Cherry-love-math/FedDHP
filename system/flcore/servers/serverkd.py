import copy, random, time, json, os, numpy as np
from flcore.clients.clientkd import clientKD
from flcore.servers.serverbase import Server
from threading import Thread
import torch
class FedKD(Server):
    def __init__(self, args, times):
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0
        super().__init__(args, times)
        self.param_shapes = {name: tuple(p.data.shape) for name, p in self.global_model.named_parameters()}
        self.set_slow_clients()
        self.set_clients(clientKD)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.T_start = args.T_start
        self.T_end = args.T_end
        self.energy = self.T_start
        self.compressed_param = {}

    def train(self):
        if not getattr(self, "compressed_param", None): self.decomposition()
        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap
                eval_t = time.time()
                self.evaluate()
                eval_cost += time.time() - eval_t
                self.current_round = _orig_round
            for client in self.selected_clients: client.train()
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0: self.call_dlg(i)
            self.aggregate_parameters()
            self.decomposition(current_round=self.current_round)
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
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt): break
            self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage raw time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        print("\nAverage training time per round excluding evaluation.")
        print(sum(self.Budget_train[1:]) / len(self.Budget_train[1:]))
        self.save_results(); self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientKD)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

        if hasattr(self, "tb") and self.tb is not None:
            try: self.tb.flush()
            except Exception: pass
            self.tb.close(); self.tb = None

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
            "total_round_wall_time_sec": float(sum(self.Budget)),
            "avg_round_wall_time_sec": float(np.mean(self.Budget[1:])) if len(self.Budget) > 1 else float("nan"),
            "total_train_wall_time_excl_eval_sec": float(sum(self.Budget_train)),
            "avg_train_wall_time_excl_eval_sec": float(np.mean(self.Budget_train[1:])) if len(self.Budget_train) > 1 else float("nan"),
            "total_eval_wall_time_sec": float(sum(self.Budget_eval)),
        }
        save_dir = getattr(self, "save_folder_name", ".")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "comm_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\n[Comm] Down total: {total_down / (1024*1024):.3f} MB")
        print(f"[Comm] Up   total: {total_up / (1024*1024):.3f} MB")
        print(f"[Comm] Total     : {(total_up + total_down) / (1024*1024):.3f} MB")
        print(f"[Comm] Per client: {avg_per_client / (1024*1024):.3f} MB\n")
        self.summarize_run()

    def _payload_bytes(self, param_dict):
        total = 0
        for v in param_dict.values():
            if isinstance(v, list) and len(v) == 3:
                total += v[0].nbytes + v[1].nbytes + v[2].nbytes
            else:
                total += v.nbytes
        return int(total)
   
    def send_models(self):
        assert len(self.clients) > 0
        per_client = self._payload_bytes(self.compressed_param)
        down = per_client * len(self.selected_clients)
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.compressed_param, self.energy)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        self.round_downlink_bytes.append(int(down))
        self.total_client_rounds += len(self.selected_clients)
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)

    def _recover_param(self, packed, target_shape):
        u, s, vt = packed
        mat = (u * s[np.newaxis, :]) @ vt
        return mat.reshape(target_shape)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        active_clients = random.sample(self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids = []; self.uploaded_models = []; up = 0
        for client in active_clients:
            try:
                client_time_cost = (client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] +
                                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds'])
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                if hasattr(client, "compressed_param"):
                    up += self._payload_bytes(client.compressed_param)
                for k in client.compressed_param.keys():
                    v = client.compressed_param[k]
                    if isinstance(v, list) and len(v) == 3:
                        target_shape = self.param_shapes.get(k)
                        if target_shape is not None:
                            client.compressed_param[k] = self._recover_param(v, target_shape)
                self.uploaded_models.append(client.compressed_param)
        self.round_uplink_bytes.append(int(up))
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            self.tb.add_scalar("comm/total_MB", (up + self.round_downlink_bytes[-1]) / (1024 * 1024), self.current_round)
        sel = len(self.selected_clients)
        act_sampled = len(active_clients)
        act_uploaded = len(self.uploaded_models)
        down = self.round_downlink_bytes[-1] if self.round_downlink_bytes else 0
        if self.current_round % 5 == 0:
            print(f"[CommDbg] round={self.current_round} sel={sel} act_sampled={act_sampled} "
                  f"act_uploaded={act_uploaded} down_total_MB={down/(1024*1024):.3f} "
                  f"up_total_MB={up/(1024*1024):.3f} "
                  f"down_per_client_MB={down/(1024*1024)/max(sel,1):.3f} "
                  f"up_per_uploaded_MB={up/(1024*1024)/max(act_uploaded,1):.3f}")

    def aggregate_parameters(self):
        if not self.uploaded_models: return
        avg = {}
        keys = self.uploaded_models[0].keys()
        for k in keys:
            avg[k] = np.zeros_like(self.uploaded_models[0][k], dtype=np.float32)
        w = 1.0 / len(self.uploaded_models)
        for client_sd in self.uploaded_models:
            for k in keys:
                avg[k] += client_sd[k].astype(np.float32, copy=False) * w
        torch_sd = {}
        ref_sd = self.global_model.state_dict()
        device = next(self.global_model.parameters()).device
        for k, v in avg.items():
            if k in ref_sd:
                t = torch.from_numpy(v).to(device=device, dtype=ref_sd[k].dtype)
                torch_sd[k] = t
        self.global_model.load_state_dict(torch_sd, strict=False)

    def _svd_pack_2d(self, mat2d, energy, raw_bytes=None):
        u, s, vt = np.linalg.svd(mat2d, full_matrices=False)
        u = u.astype(np.float32, copy=False)
        s = s.astype(np.float32, copy=False)
        vt = vt.astype(np.float32, copy=False)
        s2 = s * s; tot = float(s2.sum())
        if tot == 0.0: return None
        cum = np.cumsum(s2)
        r = int(np.searchsorted(cum, energy * tot) + 1)
        r = max(1, min(r, s.shape[0]))
        u, s, vt = u[:, :r], s[:r], vt[:r, :]
        if raw_bytes is not None and (u.nbytes + s.nbytes + vt.nbytes) >= raw_bytes:
            return None
        return [u, s, vt]
    def _svd_pack_2d_torch(self, mat_t, energy, raw_bytes):
        # 确保数据在 tensor 上
        if not torch.is_tensor(mat_t):
            mat_t = torch.as_tensor(mat_t, dtype=torch.float32)
        elif mat_t.dtype != torch.float32:
            mat_t = mat_t.float()

        # 利用 GPU (如果 mat_t 在 GPU 上) 进行 SVD
        with torch.no_grad():
            try:
                u, s, vh = torch.linalg.svd(mat_t, full_matrices=False)
            except RuntimeError:
                return None # 极少数不收敛情况
            s2 = s * s
            tot = float(s2.sum().item())
            if tot <= 0.0:
                return None
            cum = torch.cumsum(s2, dim=0)
            target = float(energy) * tot
            r = int(torch.searchsorted(cum, target).item() + 1)
            r = max(1, min(r, int(s.shape[0])))

            u = u[:, :r].contiguous()
            s = s[:r].contiguous()
            vh = vh[:r, :].contiguous()
        comp_bytes = (u.numel() + s.numel() + vh.numel()) * 4
        if comp_bytes >= int(raw_bytes):
            return None

        # 转换回 Numpy 以兼容你原本的通信协议
        return [
            u.detach().cpu().numpy(),
            s.detach().cpu().numpy(),
            vh.detach().cpu().numpy()
        ]
    def decomposition(self, current_round=None):
        self.compressed_param = {}
        gm = self.global_model
        
        # --- 保持原有 param_shapes 逻辑不变 ---
        if isinstance(gm, dict):
            iterator = gm.items()
            if not getattr(self, "param_shapes", None):
                self.param_shapes = {k: np.shape(v) for k, v in iterator}
                iterator = gm.items()
        else:
            sd = gm.state_dict()
            iterator = sd.items()
            if not getattr(self, "param_shapes", None):
                self.param_shapes = {k: tuple(v.shape) for k, v in sd.items()}

        # 设定计算设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for name, w in iterator:
            # --- 优化点 1: 保持在 GPU 上，不要立即转 numpy ---
            if torch.is_tensor(w):
                w_t = w.to(device, non_blocking=True)
            else:
                w_t = torch.tensor(w).to(device)

            # --- 优化点 2: 过滤逻辑 (Embedding 和 非浮点数) ---
            # 保持你的逻辑：Embedding 不压缩，非浮点不压缩
            if 'embeddings' in name or not w_t.is_floating_point():
                self.compressed_param[name] = w_t.detach().cpu().numpy()
                continue
            
            w32 = w_t.float()

            # --- 优化点 3: 维度检查 ---
            # 保持你的逻辑：维度>1 且 第一维>1 才压缩
            if not (w32.ndim > 1 and w32.shape[0] > 1):
                self.compressed_param[name] = w32.detach().cpu().numpy().astype(np.float32, copy=False)
                continue

            # --- 优化点 4: 调用 GPU SVD ---
            raw_bytes = w32.numel() * 4
            mat = w32.reshape(w32.shape[0], -1)
            
            # 使用新加的 _torch 版本函数
            packed = self._svd_pack_2d_torch(mat, self.energy, raw_bytes)
            
            if packed is None:
                self.compressed_param[name] = w32.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                self.compressed_param[name] = packed

        # --- 保持原有调试打印逻辑不变 ---
        comp_bytes = self._payload_bytes(self.compressed_param)
        if comp_bytes > 10 * 1024 * 1024:
            rs = [v[1].shape[0] for v in self.compressed_param.values() if isinstance(v, list) and len(v) == 3]
            print(f"[SVDdbg] energy={self.energy:.4f} comp_MB={comp_bytes/1024/1024:.3f} svd_layers={len(rs)} "
                  f"mean_r={np.mean(rs) if rs else 0:.2f} median_r={np.median(rs) if rs else 0:.0f} max_r={max(rs) if rs else 0}")

    def summarize_run(self):
        k = getattr(self.args, "summary_k", 5)
        out_name = getattr(self.args, "summary_out", "summary.json")
        acc = list(getattr(self, "rs_test_acc", []))
        top5 = list(getattr(self, "rs_test_acc_top5", [])) if hasattr(self, "rs_test_acc_top5") else []
        if not acc: 
            print("[Summary] rs_test_acc is empty, skip."); return
        k_eff = min(k, len(acc))
        summary = {
            "best_top1": float(np.max(acc)),
            "final_top1": float(acc[-1]),
            f"last{k_eff}_avg_top1": float(np.mean(acc[-k_eff:])),
        }
        if top5:
            k_eff5 = min(k, len(top5))
            summary.update({
                "best_top5": float(np.max(top5)),
                "final_top5": float(top5[-1]),
                f"last{k_eff5}_avg_top5": float(np.mean(top5[-k_eff5:])),
            })
        print("========== Run Summary ==========")
        for kk, vv in summary.items():
            print(f"{kk}: {vv:.4f}" if isinstance(vv, float) else f"{kk}: {vv}")
        print("================================")
        save_dir = getattr(self, "save_path", None) or getattr(self, "save_folder_name", "items")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, out_name), "w", encoding="utf-8") as f:
            json.dump({"args": vars(self.args), "summary": summary}, f, ensure_ascii=False, indent=2)

    def debug_compress_report(self, topk=20):
        items = []
        for name, raw in self.global_model.state_dict().items():
            raw32 = raw.cpu().numpy().astype(np.float32, copy=False)
            raw_b = raw32.nbytes
            comp = self.compressed_param[name]
            if isinstance(comp, list) and len(comp) == 3:
                r = comp[1].shape[0]
                comp_b = comp[0].nbytes + comp[1].nbytes + comp[2].nbytes
                kind = "svd"
            else:
                comp_b = comp.nbytes
                kind = "raw"
                r = None
            ratio = comp_b / max(1, raw_b)
            items.append((ratio, raw_b, comp_b, kind, name, raw.shape))
        items.sort(reverse=True)
        print("==== Worst compression ratios (comp/raw) ====")
        for ratio, raw_b, comp_b, kind, name, shape in items[:topk]:
            print(f"{ratio:6.2f}x  raw={raw_b/1024/1024:8.3f}MB  comp={comp_b/1024/1024:8.3f}MB  {kind:3s}  {shape}  {name}")
        print("===========================================")