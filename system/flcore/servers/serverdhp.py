import copy
import random
import time
import numpy as np
from flcore.clients.clientdhp import ClientDHP
from flcore.servers.serverbase import Server
from threading import Thread
import json, os
import torch
from pathlib import Path
class FedDHP(Server):
    def __init__(self, args, times):
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0
        super().__init__(args, times)
        self.save_folder_name = Path(self.logdir) if hasattr(self, 'logdir') else Path("items")
        self.save_folder_name.mkdir(parents=True, exist_ok=True)
        
        self.param_shapes = {}
        for name, p in self.global_model.named_parameters():
            self.param_shapes[name] = tuple(p.data.shape)
            
        self.set_slow_clients()
        self.set_clients(ClientDHP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        print(f"[FedDHP] Artifacts will be saved to: {self.save_folder_name}")
        # self.load_model()
        self.Budget = []
        self.Budget_train = []
        self.Budget_eval = []
        self.svd_decomp_time = 0.0
        self.svd_recover_time = 0.0
        self.T_start = args.T_start
        self.T_end = args.T_end
        self.energy = self.T_start
        self.compressed_param = {}
        # [New] Hyperparameter: "Budget Constraint"
        # 0.95 means: even if Energy=0.99, the compressed size MUST be < 95% of raw size.
        # This forces the SVD to cut off the "noise tail" (tiny singular values).
        self.min_compression_ratio = 0.95 
        # [New] Store stats for "Advanced Visualization" later
        self.vis_svd_stats = []
        self.global_prior = torch.ones(self.num_classes, dtype=torch.float32)
        self.global_prior = self.global_prior / self.global_prior.sum()
        self.prior_ema = float(getattr(args, "prior_ema", 0.2))
        self.privacy_audit_enabled = self._env_flag("FEDDHP_PRIVACY_AUDIT", bool(getattr(args, "privacy_audit", True)))
        self.privacy_gap = int(os.environ.get("FEDDHP_PRIVACY_GAP", getattr(args, "privacy_gap", 10)))
        if self.privacy_gap <= 0:
            self.privacy_gap = 10
        self.privacy_prior_noise_stds = self._env_float_list("FEDDHP_PRIOR_NOISE_STDS", getattr(args, "privacy_prior_noise_stds", "0,0.01,0.03,0.05,0.10"))
        # Optional training-time noisy aggregated prior.
        # Default 0.0 means the original FedDHP training is unchanged.
        # This is a DP-inspired/noise-perturbed ablation, not a formal DP guarantee.
        self.prior_train_noise_std = float(os.environ.get("FEDDHP_PRIOR_TRAIN_NOISE_STD", getattr(args, "prior_train_noise_std", 0.0)))
        self.prior_train_noise_mode = str(os.environ.get("FEDDHP_PRIOR_TRAIN_NOISE_MODE", getattr(args, "prior_train_noise_mode", "prob"))).lower()
        self.prior_train_noise_rows = []
        self.privacy_surface_rounds = []
        self.privacy_prior_rows = []
        self.conv_diag_enabled = self._env_flag("FEDDHP_CONV_DIAG", bool(getattr(args, "conv_diag", False)))
        self.conv_diag_gap = int(os.environ.get("FEDDHP_CONV_DIAG_GAP", getattr(args, "conv_diag_gap", self.eval_gap)))
        if self.conv_diag_gap <= 0:
            self.conv_diag_gap = max(1, int(self.eval_gap))
        self.convergence_rows = []
    def _sync_all_clients_for_eval(self):
        for client in self.clients:
            client.set_parameters(self.compressed_param, self.energy, global_prior=self.global_prior)
    def train(self):
        if not getattr(self, "compressed_param", None):
            self.decomposition()

        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            eval_cost = 0.0
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                self._sync_all_clients_for_eval()
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap  # 0,1,2,... 让 test 曲线横轴回到 0~24
                eval_t = time.time()
                self.evaluate()
                eval_cost += time.time() - eval_t
                self.current_round = _orig_round       
                if self.conv_diag_enabled and i % self.conv_diag_gap == 0:
                    diag_t = time.time()
                    self._record_global_bias(i)
                    eval_cost += time.time() - diag_t

            for client in self.selected_clients:
                client.train()
            total_counts = None
            total_seen = 0
            prior_records = []
            for c in self.selected_clients:
                counts = getattr(c, "local_class_counts", None)
                seen = int(getattr(c, "local_seen", 0))
                if counts is None or seen <= 0:
                    continue
                counts_cpu = counts.detach().cpu().to(torch.float32)
                prior_records.append((int(c.id), counts_cpu.clone(), int(seen)))
                if total_counts is None:
                    total_counts = counts_cpu.clone()
                else:
                    total_counts += counts_cpu
                total_seen += seen
            if total_counts is not None and total_seen > 0:
                p_exact = torch.clamp(total_counts, min=1e-6)
                p_exact = p_exact / p_exact.sum()
                p_train = self._apply_training_prior_noise(p_exact, total_counts=total_counts)
                self.global_prior = (1.0 - self.prior_ema) * self.global_prior + self.prior_ema * p_train
                self.global_prior = torch.clamp(self.global_prior, min=1e-6)
                self.global_prior = self.global_prior / self.global_prior.sum()
                if self.privacy_audit_enabled and i % self.privacy_gap == 0:
                    self._record_training_prior_noise(i, p_exact, p_train)
            if self.privacy_audit_enabled and i % self.privacy_gap == 0:
                self._record_prior_audit(i, prior_records)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.conv_diag_enabled and i % self.conv_diag_gap == 0:
                self._record_update_disagreement(i)
            if self.privacy_audit_enabled and i % self.privacy_gap == 0:
                self._record_surface_round(i)
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
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)
            if i % 10 == 0:
                self._save_vis_data()
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage raw time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        print("\nAverage training time per round excluding evaluation.")
        print(sum(self.Budget_train[1:]) / len(self.Budget_train[1:]))
        

        if bool(getattr(self.args, "save_cross_heatmap", True)):
            try:
                self._save_cross_client_heatmaps(tag="final")
            except Exception as e:
                print(f"[Heatmap] skipped due to error: {e}")

        self.save_results()
        self.save_global_model()
        self.save_comm_stats() 
        if self.conv_diag_enabled:
            self.save_convergence_diagnostics()
        if self.privacy_audit_enabled:
            self.save_privacy_tables()
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(ClientDHP)
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
        
        self.summarize_run()


    def _env_flag(self, name, default=False):
        value = os.environ.get(name, None)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_float_list(self, name, default):
        value = os.environ.get(name, None)
        if value is None:
            value = default
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        parts = str(value).replace(";", ",").split(",")
        out = []
        for p in parts:
            p = p.strip()
            if p:
                out.append(float(p))
        if not out:
            out = [0.0, 0.01, 0.03, 0.05, 0.10]
        return out

    def _model_param_bytes(self):
        total = 0
        for _, p in self.global_model.named_parameters():
            total += int(p.detach().numel() * p.detach().element_size())
        return int(total)

    def _mb(self, value):
        return float(value) / float(1024 * 1024)

    def _topk_overlap(self, p, q, k=3):
        k = min(int(k), int(p.numel()))
        p_top = set(torch.topk(p, k=k).indices.cpu().tolist())
        q_top = set(torch.topk(q, k=k).indices.cpu().tolist())
        return float(len(p_top.intersection(q_top))) / float(max(1, k))

    def _apply_training_prior_noise(self, p_exact, total_counts=None):
        """Return the prior used for training.

        By default this is exactly p_exact, so the original algorithm is unchanged.
        When FEDDHP_PRIOR_TRAIN_NOISE_STD > 0, we perturb the aggregated prior
        before the EMA update. This is a lightweight ablation for utility--leakage
        trade-off, not a formal differential privacy mechanism.
        """
        std = float(getattr(self, "prior_train_noise_std", 0.0))
        if std <= 0.0:
            return p_exact.clone()

        mode = str(getattr(self, "prior_train_noise_mode", "prob")).lower()
        if mode == "count" and total_counts is not None:
            # Count-space perturbation. The std is scaled by the average class count
            # to keep the command-line value interpretable across datasets.
            counts = total_counts.detach().cpu().to(torch.float32)
            avg_count = float(torch.clamp(counts.sum() / max(1, counts.numel()), min=1.0).item())
            noisy_counts = counts + torch.randn_like(counts) * std * avg_count
            noisy_counts = torch.clamp(noisy_counts, min=1e-6)
            q = noisy_counts / torch.clamp(noisy_counts.sum(), min=1e-12)
        else:
            # Probability-space perturbation. This matches the post-hoc audit scale.
            q = p_exact.detach().cpu().to(torch.float32) + torch.randn_like(p_exact.detach().cpu().to(torch.float32)) * std
            q = torch.clamp(q, min=1e-9)
            q = q / torch.clamp(q.sum(), min=1e-12)

        return q.to(dtype=p_exact.dtype)

    def _record_training_prior_noise(self, round_id, p_exact, p_train):
        if p_exact is None or p_train is None:
            return
        p = p_exact.detach().cpu().to(torch.float32)
        q = p_train.detach().cpu().to(torch.float32)
        if float(p.sum().item()) <= 0.0 or float(q.sum().item()) <= 0.0:
            return
        p = torch.clamp(p, min=1e-9); p = p / torch.clamp(p.sum(), min=1e-12)
        q = torch.clamp(q, min=1e-9); q = q / torch.clamp(q.sum(), min=1e-12)
        self.prior_train_noise_rows.append({
            "round": int(round_id),
            "train_noise_std": float(getattr(self, "prior_train_noise_std", 0.0)),
            "train_noise_mode": str(getattr(self, "prior_train_noise_mode", "prob")),
            "aggregate_l1_error": float(torch.sum(torch.abs(p - q)).item()),
            "aggregate_top1_match": float(int(torch.argmax(p).item() == torch.argmax(q).item())),
            "aggregate_top3_overlap": float(self._topk_overlap(p, q, k=3)),
            "training_affected": "Yes" if float(getattr(self, "prior_train_noise_std", 0.0)) > 0.0 else "No",
            "note": "training-time noisy aggregated prior" if float(getattr(self, "prior_train_noise_std", 0.0)) > 0.0 else "exact aggregated prior",
        })

    def _record_surface_round(self, round_id):
        if not getattr(self, "privacy_audit_enabled", False):
            return
        compressed_bytes = int(self._payload_bytes(self.compressed_param)) if getattr(self, "compressed_param", None) else 0
        full_bytes = int(self._model_param_bytes())
        uploaded_clients = int(len(getattr(self, "uploaded_models", [])))
        selected_clients = int(len(getattr(self, "selected_clients", [])))
        ratio = float(compressed_bytes / max(1, full_bytes))
        self.privacy_surface_rounds.append({
            "round": int(round_id),
            "selected": selected_clients,
            "uploaded": uploaded_clients,
            "server_seen": "SVD-compressed student/global parameters + aggregated class-prior statistics",
            "raw_data": "No",
            "raw_gradient": "No",
            "full_update": "No",
            "mentor_branch": "No",
            "class_prior": "Aggregated exact prior in default code",
            "payload_mb": self._mb(compressed_bytes),
            "full_update_ref_mb": self._mb(full_bytes),
            "payload_ref_ratio": ratio,
        })

    def _surface_summary_rows(self):
        compressed_bytes = int(self._payload_bytes(self.compressed_param)) if getattr(self, "compressed_param", None) else 0
        full_bytes = int(self._model_param_bytes())
        ratio = float(compressed_bytes / max(1, full_bytes))
        return [
            {
                "setting": "raw-gradient worst case",
                "raw_data": "No",
                "raw_gradient": "Yes",
                "full_update": "No",
                "student_global": "Gradient of current model",
                "mentor_branch": "No",
                "class_prior": "No",
                "server_seen": "Per-batch gradients",
                "risk_focus": "Gradient inversion can reconstruct samples under strong assumptions",
                "payload_ref_ratio": "N/A",
            },
            {
                "setting": "full-update baseline",
                "raw_data": "No",
                "raw_gradient": "No",
                "full_update": "Yes",
                "student_global": "Full local model/update",
                "mentor_branch": "Depends on method",
                "class_prior": "No",
                "server_seen": "Complete client update after local training",
                "risk_focus": "Model/update inversion and inference risks",
                "payload_ref_ratio": 1.0,
            },
            {
                "setting": "ours default",
                "raw_data": "No",
                "raw_gradient": "No",
                "full_update": "No",
                "student_global": "SVD-compressed parameters",
                "mentor_branch": "No",
                "class_prior": "Aggregated exact prior in current code",
                "server_seen": "Compressed student/global branch and class-prior statistics",
                "risk_focus": "Reduced observable branch; prior statistics may reveal label distribution",
                "payload_ref_ratio": ratio,
            },
            {
                "setting": "ours noisy-prior extension",
                "raw_data": "No",
                "raw_gradient": "No",
                "full_update": "No",
                "student_global": "SVD-compressed parameters",
                "mentor_branch": "No",
                "class_prior": "Noisy prior statistics",
                "server_seen": "Compressed student/global branch and perturbed class prior",
                "risk_focus": "Prior recoverability is reduced by perturbing local class statistics",
                "payload_ref_ratio": ratio,
            },
        ]

    def _record_prior_audit(self, round_id, prior_records):
        if not prior_records:
            return
        for noise_std in self.privacy_prior_noise_stds:
            l1_values = []
            top1_values = []
            top3_values = []
            for cid, counts, seen in prior_records:
                counts = counts.to(torch.float32)
                if float(counts.sum().item()) <= 0.0:
                    continue
                p = torch.clamp(counts, min=0.0)
                p = p / torch.clamp(p.sum(), min=1e-12)
                if float(noise_std) > 0.0:
                    q = p + torch.randn_like(p) * float(noise_std)
                    q = torch.clamp(q, min=1e-9)
                    q = q / torch.clamp(q.sum(), min=1e-12)
                else:
                    q = p.clone()
                l1_values.append(float(torch.sum(torch.abs(p - q)).item()))
                top1_values.append(float(int(torch.argmax(p).item() == torch.argmax(q).item())))
                k = min(3, int(p.numel()))
                p_top = set(torch.topk(p, k=k).indices.cpu().tolist())
                q_top = set(torch.topk(q, k=k).indices.cpu().tolist())
                top3_values.append(float(len(p_top.intersection(q_top))) / float(max(1, k)))
            if not l1_values:
                continue
            self.privacy_prior_rows.append({
                "round": int(round_id),
                "noise_std": float(noise_std),
                "clients": int(len(l1_values)),
                "mean_l1_error": float(np.mean(l1_values)),
                "dominant_label_recovery": float(np.mean(top1_values)),
                "top3_overlap": float(np.mean(top3_values)),
                "training_affected": "No",
                "mode": "post-hoc audit",
            })

    def _write_table(self, path_base, rows):
        if not rows:
            return
        path_base = Path(path_base)
        path_base.parent.mkdir(parents=True, exist_ok=True)
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        tsv_path = path_base.with_suffix(".tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("\t".join(keys) + "\n")
            for row in rows:
                values = []
                for key in keys:
                    value = row.get(key, "")
                    if isinstance(value, float):
                        values.append(f"{value:.8g}")
                    else:
                        values.append(str(value).replace("\t", " ").replace("\n", " "))
                f.write("\t".join(values) + "\n")
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = path_base.name[:31]
            ws.append(keys)
            for row in rows:
                ws.append([row.get(key, "") for key in keys])
            for column_cells in ws.columns:
                max_len = 0
                col = column_cells[0].column_letter
                for cell in column_cells:
                    max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                ws.column_dimensions[col].width = min(max(max_len + 2, 10), 60)
            wb.save(path_base.with_suffix(".xlsx"))
        except Exception as e:
            print(f"[Privacy] xlsx export skipped for {path_base.name}: {e}")
    def _get_conv_row(self, round_id):
        for row in self.convergence_rows:
            if int(row.get("round", -1)) == int(round_id):
                return row
        row = {"round": int(round_id)}
        self.convergence_rows.append(row)
        return row
    def _record_global_bias(self, round_id):
        if getattr(self, "global_test_loader", None) is None:
            return
        model = self.global_model
        prev_mode = model.training
        model.eval()
        device = getattr(self, "device", None)
        if device is None:
            device = next(model.parameters()).device
        c = int(self.num_classes)
        true_counts = torch.zeros(c, dtype=torch.float64)
        pred_counts = torch.zeros(c, dtype=torch.float64)
        correct_counts = torch.zeros(c, dtype=torch.float64)
        with torch.no_grad():
            for x, y in self.global_test_loader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).long().view(-1)
                logits = model(x)
                pred = torch.argmax(logits, dim=1).long().view(-1)
                true_counts += torch.bincount(y.detach().cpu(), minlength=c).to(torch.float64)
                pred_counts += torch.bincount(pred.detach().cpu(), minlength=c).to(torch.float64)
                ok = pred.eq(y)
                if ok.any():
                    correct_counts += torch.bincount(y[ok].detach().cpu(), minlength=c).to(torch.float64)
        total_true = torch.clamp(true_counts.sum(), min=1.0)
        total_pred = torch.clamp(pred_counts.sum(), min=1.0)
        p_true = true_counts / total_true
        p_pred = pred_counts / total_pred
        pred_l1 = float(torch.sum(torch.abs(p_pred - p_true)).item())
        pred_linf = float(torch.max(torch.abs(p_pred - p_true)).item())
        valid = true_counts > 0
        per_class_acc = torch.zeros_like(true_counts)
        per_class_acc[valid] = correct_counts[valid] / torch.clamp(true_counts[valid], min=1.0)
        acc_valid = per_class_acc[valid]
        if acc_valid.numel() > 0:
            acc_mean = float(acc_valid.mean().item())
            acc_std = float(torch.sqrt(torch.mean((acc_valid - acc_valid.mean()) ** 2)).item())
            acc_min = float(acc_valid.min().item())
            acc_max = float(acc_valid.max().item())
        else:
            acc_mean = float("nan")
            acc_std = float("nan")
            acc_min = float("nan")
            acc_max = float("nan")
        row = self._get_conv_row(round_id)
        row["global_pred_l1_bias"] = pred_l1
        row["global_pred_linf_bias"] = pred_linf
        row["per_class_acc_std"] = acc_std
        row["per_class_acc_mean"] = acc_mean
        row["per_class_acc_min"] = acc_min
        row["per_class_acc_max"] = acc_max
        row["global_bias_total"] = int(total_true.item())
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("convergence/global_pred_l1_bias", pred_l1, int(round_id))
            self.tb.add_scalar("convergence/global_pred_linf_bias", pred_linf, int(round_id))
            self.tb.add_scalar("convergence/per_class_acc_std", acc_std, int(round_id))
            self.tb.add_scalar("convergence/per_class_acc_min", acc_min, int(round_id))
        model.train(prev_mode)
    def _global_param_flat_cache(self):
        chunks = []
        keys = []
        for name, p in self.global_model.named_parameters():
            arr = p.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
            chunks.append(arr)
            keys.append(name)
        if not chunks:
            return [], np.zeros(0, dtype=np.float32)
        return keys, np.concatenate(chunks).astype(np.float32, copy=False)
    def _client_param_flat(self, client_sd, keys):
        chunks = []
        for k in keys:
            if k not in client_sd:
                return None
            v = client_sd[k]
            if isinstance(v, list) and len(v) == 3:
                return None
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            chunks.append(arr)
        if not chunks:
            return None
        return np.concatenate(chunks).astype(np.float32, copy=False)
    def _record_update_disagreement(self, round_id):
        if not getattr(self, "uploaded_models", None):
            return
        keys, global_flat = self._global_param_flat_cache()
        if global_flat.size == 0:
            return
        flats = []
        for client_sd in self.uploaded_models:
            flat = self._client_param_flat(client_sd, keys)
            if flat is not None and flat.shape == global_flat.shape:
                flats.append(flat)
        n = len(flats)
        if n <= 0:
            return
        mean_delta = np.zeros_like(global_flat, dtype=np.float32)
        update_norms = []
        for flat in flats:
            delta = flat - global_flat
            mean_delta += delta / float(n)
            update_norms.append(float(np.linalg.norm(delta)))
        disagreement = 0.0
        cosine_values = []
        mean_norm = float(np.linalg.norm(mean_delta))
        for flat in flats:
            delta = flat - global_flat
            diff = delta - mean_delta
            disagreement += float(np.dot(diff, diff)) / float(n)
            dn = float(np.linalg.norm(delta))
            if dn > 1e-12 and mean_norm > 1e-12:
                cosine_values.append(float(np.dot(delta, mean_delta) / (dn * mean_norm)))
        update_norm_mean = float(np.mean(update_norms)) if update_norms else float("nan")
        update_norm_std = float(np.std(update_norms)) if update_norms else float("nan")
        cosine_mean = float(np.mean(cosine_values)) if cosine_values else float("nan")
        row = self._get_conv_row(round_id)
        row["uploaded_clients"] = int(n)
        row["update_norm_mean"] = update_norm_mean
        row["update_norm_std"] = update_norm_std
        row["update_disagreement"] = float(disagreement)
        row["mean_update_norm"] = mean_norm
        row["mean_update_cosine"] = cosine_mean
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("convergence/update_disagreement", float(disagreement), int(round_id))
            self.tb.add_scalar("convergence/update_norm_mean", update_norm_mean, int(round_id))
            self.tb.add_scalar("convergence/update_norm_std", update_norm_std, int(round_id))
            self.tb.add_scalar("convergence/mean_update_norm", mean_norm, int(round_id))
            if np.isfinite(cosine_mean):
                self.tb.add_scalar("convergence/mean_update_cosine", cosine_mean, int(round_id))
    def save_convergence_diagnostics(self):
        out_dir = Path(getattr(self, "save_folder_name", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "_write_table"):
            self._write_table(out_dir / "convergence_diagnostics", self.convergence_rows)
        else:
            path = out_dir / "convergence_diagnostics.tsv"
            if not self.convergence_rows:
                return
            keys = []
            for row in self.convergence_rows:
                for k in row.keys():
                    if k not in keys:
                        keys.append(k)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\t".join(keys) + "\n")
                for row in self.convergence_rows:
                    f.write("\t".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"[ConvDiag] saved to {out_dir}")
    def save_privacy_tables(self):
        out_dir = Path(getattr(self, "save_folder_name", "."))
        self._write_table(out_dir / "privacy_surface", self._surface_summary_rows())
        self._write_table(out_dir / "privacy_surface_rounds", self.privacy_surface_rounds)
        self._write_table(out_dir / "privacy_prior_leakage", self.privacy_prior_rows)
        self._write_table(out_dir / "privacy_train_noisy_prior", self.prior_train_noise_rows)
        print(f"[Privacy] tables saved to {out_dir}")

    def _save_cross_client_heatmaps(self, tag="final"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = Path(getattr(self, "save_folder_name", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        clients = list(getattr(self, "clients", []))
        n = len(clients)
        if n == 0:
            return
        has_teacher = all(hasattr(c, "model") for c in clients)
        has_student = all(hasattr(c, "global_model") for c in clients)
        if not has_teacher or not has_student:
            raise RuntimeError("clients must have both .model (teacher) and .global_model (student)")
        teacher_models = [c.model for c in clients]
        student_models = [c.global_model for c in clients]
        teacher_prev_modes = [m.training for m in teacher_models]
        student_prev_modes = [m.training for m in student_models]
        for m in teacher_models:
            m.eval()
        for m in student_models:
            m.eval()
        loaders = [c.load_test_data() for c in clients]
        teacher_correct = np.zeros((n, n), dtype=np.float64)
        student_correct = np.zeros((n, n), dtype=np.float64)
        totals = np.zeros(n, dtype=np.float64)
        with torch.no_grad():
            device = getattr(self, "device", None)
            if device is None:
                device = next(teacher_models[0].parameters()).device
            for m in teacher_models:
                if next(m.parameters()).device != device:
                    m.to(device)
            for m in student_models:
                if next(m.parameters()).device != device:
                    m.to(device)
            for j in range(n):
                loader = loaders[j]
                for x, y in loader:
                    bs = int(y.shape[0])
                    totals[j] += bs
                    x_d = x.to(device, non_blocking=True)
                    y_d = y.to(device, non_blocking=True)
                    for i in range(n):
                        out_t = teacher_models[i](x_d)
                        pred_t = out_t.argmax(dim=1)
                        teacher_correct[i, j] += float((pred_t == y_d).sum().item())
                        out_s = student_models[i](x_d)
                        pred_s = out_s.argmax(dim=1)
                        student_correct[i, j] += float((pred_s == y_d).sum().item())
        totals = np.maximum(totals, 1.0)
        teacher_acc = (teacher_correct / totals[np.newaxis, :]) * 100.0
        student_acc = (student_correct / totals[np.newaxis, :]) * 100.0
        np.save(out_dir / f"heatmap_local_teacher_{tag}.npy", teacher_acc)
        np.save(out_dir / f"heatmap_global_student_{tag}.npy", student_acc)
        np.savetxt(out_dir / f"heatmap_local_teacher_{tag}.csv", teacher_acc, delimiter=",", fmt="%.6f")
        np.savetxt(out_dir / f"heatmap_global_student_{tag}.csv", student_acc, delimiter=",", fmt="%.6f")
        def _plot(mat, title, fname):
            fig = plt.figure(figsize=(8, 6), dpi=200)
            ax = fig.add_subplot(111)
            im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=100.0, cmap="coolwarm")
            ax.set_xlabel("Local test set ID")
            ax.set_ylabel("Client model ID")
            ax.set_title(title)
            ax.set_xticks(list(range(n)))
            ax.set_yticks(list(range(n)))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy (%)")
            fig.tight_layout()
            fig.savefig(out_dir / fname, bbox_inches="tight")
            plt.close(fig)
        _plot(teacher_acc, "Local Teacher cross-client test accuracy (%)", f"heatmap_local_teacher_{tag}.png")
        _plot(student_acc, "Global Student cross-client test accuracy (%)", f"heatmap_global_student_{tag}.png")
        for m, prev in zip(teacher_models, teacher_prev_modes):
            m.train(prev)
        for m, prev in zip(student_models, student_prev_modes):
            m.train(prev)

    def _payload_bytes(self, param_dict):
        total = 0
        for v in param_dict.values():
            if isinstance(v, list) and len(v) == 3:
                total += v[0].nbytes + v[1].nbytes + v[2].nbytes
            else:
                total += v.nbytes
        return int(total)

    def send_models(self):
        assert (len(self.clients) > 0)
        per_client = self._payload_bytes(self.compressed_param)
        down = per_client * len(self.selected_clients)
        for client in self.selected_clients:
            start_time = time.time()
            
            client.set_parameters(self.compressed_param, self.energy, global_prior=self.global_prior)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        self.round_downlink_bytes.append(int(down))
        self.total_client_rounds += int(len(self.selected_clients))
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)

    def _recover_param(self, packed, target_shape):
        u, s, vt = packed
        mat = (u * s[np.newaxis, :]) @ vt
        return mat.reshape(target_shape)
    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_weights = []
        up = 0
        recover_cost = 0.0
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                n = getattr(client, "train_samples", 1) # 获取样本数，默认 1
                self.uploaded_weights.append(n)         # 暂时存入原始数量
                tot_samples += n                        # 累加总数

                if hasattr(client, "compressed_param"):
                    up += self._payload_bytes(client.compressed_param)

                for k in client.compressed_param.keys():
                    v = client.compressed_param[k]
                    if isinstance(v, list) and len(v) == 3:
                        target_shape = self.param_shapes.get(k, None)
                        if target_shape is None:
                            continue
                        rt = time.time()
                        client.compressed_param[k] = self._recover_param(v, target_shape)
                        recover_cost += time.time() - rt
                self.uploaded_models.append(client.compressed_param)
  #      if tot_samples > 0:
  #          self.uploaded_weights = [w / tot_samples for w in self.uploaded_weights]
        self.round_uplink_bytes.append(int(up))
        self.svd_recover_time += recover_cost
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("time/svd_recover_sec", recover_cost, self.current_round)
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            self.tb.add_scalar("comm/total_MB", (up + self.round_downlink_bytes[-1]) / (1024 * 1024), self.current_round)
        sel = len(self.selected_clients)
        act_sampled = len(active_clients)              # 掉线模拟前/后？这里是 sample 后（drop_rate 后）
        act_uploaded = len(self.uploaded_models)        # 真正“算进 uplink”的人数（过 time_threshold 的）
        down = int(self.round_downlink_bytes[-1]) if len(self.round_downlink_bytes) else 0
        if self.current_round % 5 == 0:
            print(
                f"[CommDbg] round={self.current_round} sel={sel} "
                f"act_sampled={act_sampled} act_uploaded={act_uploaded} "
                f"down_total_MB={down/(1024*1024):.3f} up_total_MB={up/(1024*1024):.3f} "
                f"down_per_client_MB={down/(1024*1024)/max(sel,1):.3f} "
                f"up_per_uploaded_MB={up/(1024*1024)/max(act_uploaded,1):.3f}"
            )
    def aggregate_parameters(self):
        if len(self.uploaded_models) == 0:
            return

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
                t = torch.from_numpy(v)
                t = t.to(device=device, dtype=ref_sd[k].dtype)
                torch_sd[k] = t

        self.global_model.load_state_dict(torch_sd, strict=False)
#     def aggregate_parameters(self):
#         if len(self.uploaded_models) == 0:
#             return

#         avg = {}
#         keys = self.uploaded_models[0].keys()
#         for k in keys:
#             avg[k] = np.zeros_like(self.uploaded_models[0][k], dtype=np.float32)

#         weights = getattr(self, "uploaded_weights", None)
#         if weights is None or len(weights) != len(self.uploaded_models):
#             weights = [1.0 / len(self.uploaded_models) for _ in self.uploaded_models]
#         else:
#             s = float(sum(weights))
#             if s <= 0.0:
#                 weights = [1.0 / len(self.uploaded_models) for _ in self.uploaded_models]
#             else:
#                 weights = [float(w) / s for w in weights]

#         for w, client_sd in zip(weights, self.uploaded_models):
#             for k in keys:
#                 avg[k] += client_sd[k].astype(np.float32, copy=False) * float(w)

#         torch_sd = {}
#         ref_sd = self.global_model.state_dict()
#         device = next(self.global_model.parameters()).device

#         for k, v in avg.items():
#             if k in ref_sd:
#                 t = torch.from_numpy(v)
#                 t = t.to(device=device, dtype=ref_sd[k].dtype)
#                 torch_sd[k] = t

#         self.global_model.load_state_dict(torch_sd, strict=False)

#     def save_comm_stats(self):
#         total_up = int(sum(self.round_uplink_bytes))
#         total_down = int(sum(self.round_downlink_bytes))
#         avg_per_client = (total_up + total_down) / max(1, self.num_clients)
#         stats = {
#             "total_uplink_bytes": total_up,
#             "total_downlink_bytes": total_down,
#             "total_bytes": total_up + total_down,
#             "avg_bytes_per_client": float(avg_per_client),
#         }
#       # 使用 self.save_folder_name (即带有时间戳的 logdir)
#         save_path = self.save_folder_name / "comm_stats.json"
        
#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(stats, f, indent=2)
#         print(f"[Stats] Communication stats saved to {save_path}")
    def save_comm_stats(self):
        total_up = int(sum(self.round_uplink_bytes))
        total_down = int(sum(self.round_downlink_bytes))
        total_bytes = total_up + total_down

        avg_per_client_round = total_bytes / max(1, self.total_client_rounds)
        avg_per_client = total_bytes / max(1, self.num_clients)

        budget_train = getattr(self, "Budget_train", [])
        budget_eval = getattr(self, "Budget_eval", [])
        budget_raw = getattr(self, "Budget", [])

        stats = {
            "total_uplink_bytes": total_up,
            "total_downlink_bytes": total_down,
            "total_bytes": total_bytes,
            "total_client_rounds": int(self.total_client_rounds),
            "avg_bytes_per_client_round": float(avg_per_client_round),
            "avg_bytes_per_client": float(avg_per_client),

            "total_round_wall_time_sec": float(sum(budget_raw)),
            "avg_round_wall_time_sec": float(np.mean(budget_raw[1:])) if len(budget_raw) > 1 else float("nan"),

            "total_train_wall_time_excl_eval_sec": float(sum(budget_train)),
            "avg_train_wall_time_excl_eval_sec": float(np.mean(budget_train[1:])) if len(budget_train) > 1 else float("nan"),

            "total_eval_wall_time_sec": float(sum(budget_eval)),
            "total_svd_decomp_time_sec": float(getattr(self, "svd_decomp_time", 0.0)),
            "total_svd_recover_time_sec": float(getattr(self, "svd_recover_time", 0.0)),
        }

        save_path = self.save_folder_name / "comm_stats.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"[Stats] Efficiency stats saved to {save_path}")
        print(f"[Comm] Down total: {total_down / (1024*1024):.3f} MB")
        print(f"[Comm] Up   total: {total_up / (1024*1024):.3f} MB")
        print(f"[Comm] Total     : {total_bytes / (1024*1024):.3f} MB")
        print(f"[Time] Train excl. eval: {stats['total_train_wall_time_excl_eval_sec']:.3f} s")
        print(f"[Time] SVD decomp total: {stats['total_svd_decomp_time_sec']:.3f} s")
        print(f"[Time] SVD recover total: {stats['total_svd_recover_time_sec']:.3f} s")
    def _save_vis_data(self):
        """Dump SVD stats for advanced plotting later"""
        save_path = self.save_folder_name / "svd_vis_stats.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.vis_svd_stats, f, indent=2)
    def add_parameters(self, w, client_model):
        for server_k, client_k in zip(self.global_model.keys(), client_model.keys()):
            self.global_model[server_k] += client_model[client_k] * w
    
    def _svd_pack_2d(self, mat2d, energy, raw_bytes=None, min_compression_ratio=1.0):
        if not torch.is_tensor(mat2d):
            mat_t = torch.as_tensor(np.asarray(mat2d), dtype=torch.float32)
        else:
            mat_t = mat2d
            if mat_t.dtype != torch.float32:
                mat_t = mat_t.float()

        m, n = mat_t.shape
        if raw_bytes is None:
            raw_bytes = int(m * n * 4)

        max_allowed_floats = (raw_bytes / 4.0) * float(min_compression_ratio)
        r_budget = int(max_allowed_floats // (m + n + 1))
        full_rank = int(min(m, n))
        if r_budget >= full_rank:
            return None

        if torch.cuda.is_available():
            mat_t = mat_t.to('cuda', non_blocking=True)

        with torch.no_grad():
            u, s, vh = torch.linalg.svd(mat_t, full_matrices=False)
            s2 = s * s
            tot = float(s2.sum().item())
            if tot == 0.0:
                return None
            cum = torch.cumsum(s2, dim=0)
            r_energy = int(torch.searchsorted(cum, energy * tot).item() + 1)
            r_final = max(1, min(r_energy, r_budget, s.shape[0]))
            u = u[:, :r_final].contiguous()
            s = s[:r_final].contiguous()
            vh = vh[:r_final, :].contiguous()

        u_np = u.float().cpu().numpy()
        s_np = s.float().cpu().numpy()
        vt_np = vh.float().cpu().numpy()
        comp_bytes = u_np.nbytes + s_np.nbytes + vt_np.nbytes
        if comp_bytes >= raw_bytes:
            return None # Revert to raw if no saving (e.g., matrix too small)

        return [u_np, s_np, vt_np]

    def decomposition(self, current_round=None):
        t0 = time.time()
        self.compressed_param = {}
        layer_stats = []
        sd = self.global_model.state_dict()
        if not self.param_shapes:
            self.param_shapes = {k: tuple(v.shape) for k, v in sd.items()}

        device_svd = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        for name, w in sd.items():
            if not torch.is_tensor(w):
                w_t = torch.as_tensor(np.asarray(w), dtype=torch.float32, device=device_svd)
            else:
                w_t = w.detach()
                if w_t.dtype != torch.float32:
                    w_t = w_t.float()
                w_t = w_t.to(device_svd, non_blocking=True)

            if (not torch.is_floating_point(w_t)) or ("embeddings" in name) or (w_t.ndim <= 1) or (w_t.shape[0] <= 1):
                self.compressed_param[name] = w_t.detach().cpu().numpy().astype(np.float32, copy=False)
                continue

            raw_bytes = int(w_t.numel() * 4)

            if w_t.ndim == 4:
                mat_t = w_t.reshape(w_t.shape[0], -1)
            else:
                mat_t = w_t.reshape(w_t.shape[0], -1)

            packed = self._svd_pack_2d(
                mat_t,
                self.energy,
                raw_bytes=raw_bytes,
                min_compression_ratio=self.min_compression_ratio
            )

            if packed is None:
                self.compressed_param[name] = w_t.detach().cpu().numpy().astype(np.float32, copy=False)
                layer_stats.append({
                    "name": name,
                    "r": int(min(mat_t.shape)),
                    "full_rank": int(min(mat_t.shape)),
                    "ratio": 1.0
                })
            else:
                self.compressed_param[name] = packed
                r_used = int(packed[1].shape[0])
                comp_bytes = self._payload_bytes({name: packed})
                layer_stats.append({
                    "name": name,
                    "r": r_used,
                    "full_rank": int(min(mat_t.shape)),
                    "ratio": float(comp_bytes / raw_bytes)
                })

        self.vis_svd_stats.append({
            "round": current_round,
            "energy": float(self.energy),
            "layers": layer_stats
        })
        dt = time.time() - t0
        self.svd_decomp_time += dt
        if hasattr(self, "tb") and self.tb is not None and current_round is not None:
            self.tb.add_scalar("time/svd_decomp_sec", dt, current_round)


    def _safe_mean(x):
        return float(np.mean(x)) if len(x) else float('nan')

    def summarize_run(self):
        k = getattr(self.args, "summary_k", 5)
        out_name = getattr(self.args, "summary_out", "summary.json")

        acc = list(getattr(self, "rs_test_acc", []))
        top5 = list(getattr(self, "rs_test_acc_top5", [])) if hasattr(self, "rs_test_acc_top5") else []

        if len(acc) == 0:
            print("[Summary] rs_test_acc is empty, skip.")
            return

        k_eff = min(k, len(acc))
        summary = {
            "best_top1": float(np.max(acc)),
            "final_top1": float(acc[-1]),
            f"last{k_eff}_avg_top1": float(np.mean(acc[-k_eff:])),
            "prior_train_noise_std": float(getattr(self, "prior_train_noise_std", 0.0)),
            "prior_train_noise_mode": str(getattr(self, "prior_train_noise_mode", "prob")),
        }

        if len(top5):
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
            json.dump({
                "args": vars(self.args),
                "summary": summary,
            }, f, ensure_ascii=False, indent=2, default=str)
    def debug_compress_report(self, topk=20):
        items = []
        for name, raw in self.global_model.state_dict().items():
            raw32 = raw.detach().cpu().numpy().astype(np.float32, copy=False)
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
        print(f"... {kind:3s} r={r} {shape} {name}")

        print("==== Worst compression ratios (comp/raw) ====")
        for ratio, raw_b, comp_b, kind, name, shape in items[:topk]:
            print(f"{ratio:6.2f}x  raw={raw_b/1024/1024:8.3f}MB  comp={comp_b/1024/1024:8.3f}MB  {kind:3s}  {shape}  {name}")
        print("===========================================")
