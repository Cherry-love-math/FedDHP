import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from flcore.clients.clientbase import Client
import os, pickle, torch, itertools
import time
from torchvision import transforms
import math
class SharedBaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        rep = self.base(x)
        return self.head(rep)
class ClientDHP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.mentee_learning_rate = args.mentee_learning_rate

        if not hasattr(self, "num_classes"):
            self.num_classes = getattr(args, "num_classes", None)
       
        self.shared_base = copy.deepcopy(args.model.base).to(self.device)
        self.mentee_head = copy.deepcopy(args.model.head).to(self.device)

        if getattr(args, "teacher_model", None) is not None and hasattr(args.teacher_model, "head"):
            self.mentor_head = copy.deepcopy(args.teacher_model.head).to(self.device)
        else:
            self.mentor_head = copy.deepcopy(self.mentee_head).to(self.device)

        self.global_model = SharedBaseHeadSplit(self.shared_base, self.mentee_head).to(self.device)
        self.model = SharedBaseHeadSplit(self.shared_base, self.mentor_head).to(self.device)
      
        self.has_BatchNorm = any(isinstance(m, nn.BatchNorm2d) for m in self.shared_base.modules())

        self.loss = nn.CrossEntropyLoss()

        self.feature_dim_t = list(self.mentor_head.parameters())[0].shape[1]
        self.feature_dim_s = list(self.mentee_head.parameters())[0].shape[1]

        self.W_h = nn.Linear(self.feature_dim_s, self.feature_dim_t, bias=False).to(self.device)

        self.optimizer = torch.optim.SGD(
            [
                {"params": self.shared_base.parameters(), "lr": self.mentee_learning_rate},
                {"params": self.mentee_head.parameters(), "lr": self.mentee_learning_rate},
                {"params": self.mentor_head.parameters(), "lr": self.learning_rate},
                {"params": self.W_h.parameters(), "lr": self.learning_rate},
            ],
        )

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma,
        )

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()
        self.asd_T = 1.0
        self.asd_conf = 0.0
        self.compressed_param = {}
        self.energy = None

        # --- 新增参数与初始化 (Patch内容) ---
        self.yoyo_tau = float(getattr(args, "yoyo_tau", 1.0))
        self.yoyo_gamma = float(getattr(args, "yoyo_gamma", 0.5))
        self.yoyo_prior_batches = int(getattr(args, "yoyo_prior_batches", 20))
        self.asd_beta = float(getattr(args, "asd_beta", 1.0))
        self.asd_gamma = float(getattr(args, "asd_gamma", 1.0))
        self.prox_mu = float(getattr(args, "mu", 0.0))

        # --- ablation switches (args flags) ---
        self.enable_yoyo = not bool(getattr(args, "ablate_yoyo", False))
        self.enable_asd_logits = not bool(getattr(args, "ablate_asd", False))
        self.enable_asd_feat = not bool(getattr(args, "ablate_feat", False))
        self.enable_asd_mask = not bool(getattr(args, "ablate_mask", False))
        self.enable_strong_aug = not bool(getattr(args, "ablate_strong_aug", False))
        self.student_aug = str(getattr(args, "student_aug", "hflip")).lower().replace("-", "_")
        if self.student_aug in ["none", "no", "off", "raw"]:
            self.student_aug = "identity"
        if self.student_aug == "default":
            self.student_aug = "hflip"
        self.valid_cifar_student_aug = {
            "identity",
            "hflip",
            "hflip_cutout",
            "crop_hflip_cutout",
        }
        if self.student_aug not in self.valid_cifar_student_aug:
            raise ValueError(f"Unsupported student_aug={self.student_aug}. Valid choices: {sorted(self.valid_cifar_student_aug)}")
        data_name = ""
        if hasattr(args, "data"):
            data_name = str(getattr(args, "data"))
        elif hasattr(args, "dataset"):
            data_name = str(getattr(args, "dataset"))
        self._data_name = data_name.lower()
        self._is_emnist = ("emnist" in self._data_name)
        self._is_femnist = ("femnist" in self._data_name) or self._is_emnist
        if self._is_femnist:
            print(f"--- [Client {self.id}] Dataset detected: FEMNIST ---")
        else:
            print(f"--- [Client {self.id}] Dataset detected: CIFAR/Other ---")
        # --- mentor EMA (for stability) ---
        self.enable_mentor_ema = bool(getattr(args, "mentor_ema", True))
        self.mentor_ema_decay = float(getattr(args, "mentor_ema_decay", 0.99))
        self.ema_teacher_distill = bool(getattr(args, "ema_teacher_distill", True))
        self.ema_teacher_eval = bool(getattr(args, "ema_teacher_eval", True))
        if self.enable_mentor_ema:
            self.ema_shared_base = copy.deepcopy(self.shared_base).to(self.device)
            self.ema_mentor_head = copy.deepcopy(self.mentor_head).to(self.device)
            for p in self.ema_shared_base.parameters():
                p.requires_grad_(False)
            for p in self.ema_mentor_head.parameters():
                p.requires_grad_(False)
            self.ema_model = SharedBaseHeadSplit(self.ema_shared_base, self.ema_mentor_head).to(self.device)
            self.ema_model.eval()
        else:
            self.ema_shared_base = None
            self.ema_mentor_head = None
            self.ema_model = None

        if self.num_classes is None:
            self.num_classes = int(getattr(args, "num_classes_fallback", 100))

        prior_g = torch.ones(self.num_classes, dtype=torch.float32)
        prior_g = prior_g / prior_g.sum()
        self.prior_global = prior_g
        self.prior_local = prior_g.clone()

        self.strong_hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.strong_erase = transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
        self.cifar_crop_padding = int(getattr(args, "cifar_crop_padding", 4))
        self.cifar_cutout_p = float(getattr(args, "cifar_cutout_p", 0.5))
        self.cifar_cutout_scale = tuple(getattr(args, "cifar_cutout_scale", (0.02, 0.15)))
        self.cifar_cutout_ratio = tuple(getattr(args, "cifar_cutout_ratio", (0.3, 3.3)))
        if (not self._is_femnist) and self.id == 0:
            print(f"--- [AugCfg] CIFAR student_aug={self.student_aug}, crop_padding={self.cifar_crop_padding}, cutout_p={self.cifar_cutout_p}, cutout_scale={self.cifar_cutout_scale} ---")
        self.prior_local = self._estimate_local_prior(self.yoyo_prior_batches)


    @torch.no_grad()
    def _ema_update(self):
        if not self.enable_mentor_ema:
            return
        d = self.mentor_ema_decay
        for ep, p in zip(self.ema_shared_base.parameters(), self.shared_base.parameters()):
            ep.data.mul_(d).add_(p.data, alpha=1.0 - d)
        for ep, p in zip(self.ema_mentor_head.parameters(), self.mentor_head.parameters()):
            ep.data.mul_(d).add_(p.data, alpha=1.0 - d)
        for eb, b in zip(self.ema_shared_base.buffers(), self.shared_base.buffers()):
            eb.data.copy_(b.data)
        for eb, b in zip(self.ema_mentor_head.buffers(), self.mentor_head.buffers()):
            eb.data.copy_(b.data)


    def _obj_nbytes(self, obj):
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            return int(np.asarray(obj[0]).nbytes + np.asarray(obj[1]).nbytes + np.asarray(obj[2]).nbytes)
        return int(np.asarray(obj).nbytes)

    def _dict_nbytes(self, d):
        s = 0
        for v in d.values():
            s += self._obj_nbytes(v)
        return int(s)

    def _strong_augment_batch(self, x):
        if x.dim() != 4:
            return x
        if getattr(self, "_is_femnist", False):
            return self._strong_augment_femnist(x)
        else:
            return self._strong_augment_cifar(x)

    def _strong_augment_cifar(self, x):
        if x.dim() != 4:
            return x
        if not self.enable_strong_aug:
            return x
        mode = getattr(self, "student_aug", "hflip")
        if mode == "identity":
            return x
        if mode == "hflip":
            return self._cifar_hflip(x)
        if mode == "hflip_cutout":
            return self._cifar_cutout(self._cifar_hflip(x))
        if mode == "crop_hflip_cutout":
            return self._cifar_cutout(self._cifar_hflip(self._cifar_random_crop(x)))
        return self._cifar_hflip(x)
    def _cifar_hflip(self, x):
        if x.dim() != 4:
            return x
        B = x.size(0)
        mask = torch.rand(B, device=x.device) < 0.5
        if not mask.any():
            return x
        y = x.clone()
        y[mask] = y[mask].flip(-1)
        return y

    def _cifar_random_crop(self, x):
        if x.dim() != 4:
            return x
        pad = int(getattr(self, "cifar_crop_padding", 4))
        if pad <= 0:
            return x
        B, C, H, W = x.shape
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        y = torch.empty_like(x)
        for i in range(B):
            yy = int(oy[i].item())
            xx = int(ox[i].item())
            y[i] = xp[i, :, yy:yy + H, xx:xx + W]
        return y

    def _cifar_cutout(self, x):
        if x.dim() != 4:
            return x
        p = float(getattr(self, "cifar_cutout_p", 0.5))
        if p <= 0.0:
            return x
        B, C, H, W = x.shape
        y = x.clone()
        idx = (torch.rand(B, device=x.device) < p).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return y
        s_min, s_max = getattr(self, "cifar_cutout_scale", (0.02, 0.15))
        r_min, r_max = getattr(self, "cifar_cutout_ratio", (0.3, 3.3))
        area = float(H * W)
        for i in idx.tolist():
            frac = float(torch.empty(1, device=x.device).uniform_(float(s_min), float(s_max)).item())
            ratio = float(torch.empty(1, device=x.device).uniform_(float(r_min), float(r_max)).item())
            target = frac * area
            hh = int(round((target * ratio) ** 0.5))
            ww = int(round((target / ratio) ** 0.5))
            hh = max(1, min(hh, H))
            ww = max(1, min(ww, W))
            y0 = int(torch.randint(0, H - hh + 1, (1,), device=x.device).item())
            x0 = int(torch.randint(0, W - ww + 1, (1,), device=x.device).item())
            y[i, :, y0:y0 + hh, x0:x0 + ww] = 0.0
        return y
    def _strong_augment_femnist(self, x):
        if x.dim() != 4:
            return x
        B, C, H, W = x.shape
        y = x
        bg = x.amin(dim=(2, 3), keepdim=True)
        if getattr(self, "_is_emnist", False):
            p_aff = 0.60
            max_deg = 4.0
            max_trans = 0.04
            max_scale = 0.05
            p_noise = 0.08
            p_cut = 0.0
        else:
            p_aff = 0.60
            max_deg = 7.0
            max_trans = 0.06
            max_scale = 0.08
            p_noise = 0.10
            p_cut = 0.08
        m_aff = (torch.rand(B, device=x.device) < p_aff).to(x.dtype)
        ang = (torch.rand(B, device=x.device) * 2.0 - 1.0) * max_deg
        ang = ang * m_aff
        ang = ang * (math.pi / 180.0)
        sc = 1.0 + (torch.rand(B, device=x.device) * 2.0 - 1.0) * max_scale
        sc = 1.0 + (sc - 1.0) * m_aff
        dx = (torch.rand(B, device=x.device) * 2.0 - 1.0) * (max_trans * float(W))
        dy = (torch.rand(B, device=x.device) * 2.0 - 1.0) * (max_trans * float(H))
        dx = dx * m_aff
        dy = dy * m_aff
        tx = 2.0 * dx / max(float(W - 1), 1.0)
        ty = 2.0 * dy / max(float(H - 1), 1.0)
        ca = torch.cos(ang) * sc
        sa = torch.sin(ang) * sc
        theta = torch.zeros((B, 2, 3), device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = ca
        theta[:, 0, 1] = -sa
        theta[:, 1, 0] = sa
        theta[:, 1, 1] = ca
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, size=y.size(), align_corners=False)
        y = F.grid_sample(y, grid, mode="nearest", padding_mode="border", align_corners=False)
        m_noise = (torch.rand(B, device=x.device) < p_noise).to(x.dtype)
        sigma = (0.003 + 0.003 * torch.rand(B, device=x.device)) * m_noise
        y = y + torch.randn_like(y) * sigma.view(B, 1, 1, 1)
        m_cut = (torch.rand(B, device=x.device) < p_cut).nonzero(as_tuple=False).view(-1)
        if m_cut.numel() > 0:
            area = float(H * W)
            for i in m_cut.tolist():
                target = (0.02 + 0.03 * float(torch.rand(1, device=x.device))) * area
                aspect = 0.6 + 1.4 * float(torch.rand(1, device=x.device))
                hh = int(round((target * aspect) ** 0.5))
                ww = int(round((target / aspect) ** 0.5))
                if hh < 1:
                    hh = 1
                if ww < 1:
                    ww = 1
                if hh >= H:
                    hh = H - 1 if H > 1 else 1
                if ww >= W:
                    ww = W - 1 if W > 1 else 1
                y0 = int(torch.randint(0, max(H - hh, 1), (1,), device=x.device).item())
                x0 = int(torch.randint(0, max(W - ww, 1), (1,), device=x.device).item())
                y[i, :, y0:y0 + hh, x0:x0 + ww] = bg[i]
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        diff = (y - x).abs().mean(dim=(1, 2, 3))
        std = x.flatten(1).std(dim=1) + 1e-6
        bad = (diff / std) > 2.0
        if bad.any():
            y[bad] = x[bad]
        return y



    def _estimate_local_prior(self, max_batches=20):
        loader = self.load_train_data()
        counts = torch.zeros(self.num_classes, dtype=torch.float64)
        seen = 0
        for _, y in itertools.islice(loader, max_batches):
            y = y.detach().cpu().to(torch.int64).view(-1)
            if y.numel() == 0:
                continue
            bc = torch.bincount(y, minlength=self.num_classes).to(torch.float64)
            counts += bc
            seen += int(y.numel())
        if seen <= 0 or float(counts.sum()) <= 0.0:
            p = torch.ones(self.num_classes, dtype=torch.float32)
            self.local_class_counts = torch.zeros(self.num_classes, dtype=torch.float32)
            self.local_seen = 0
            return p / p.sum()
        p = (counts / counts.sum()).to(torch.float32)
        p = torch.clamp(p, min=1e-6)
        p = p / p.sum()
        self.local_class_counts = counts.to(torch.float32)
        self.local_seen = int(seen)
        return p

    def _get_mixed_prior(self):
        pg = self.prior_global
        pl = self.prior_local if self.prior_local is not None else pg
        p = (1.0 - self.yoyo_gamma) * pg + self.yoyo_gamma * pl
        p = torch.clamp(p, min=1e-6)
        p = p / p.sum()
        return p

    def train(self):
        trainloader = self.load_train_data()
        
        self.model.train()
        self.global_model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        if self.enable_yoyo:
            self.prior_local = self._estimate_local_prior(self.yoyo_prior_batches)
            prior = self._get_mixed_prior().to(self.device)
            adj = self.yoyo_tau * torch.log(prior + 1e-9)
        else:
            adj = torch.zeros(self.num_classes, device=self.device)
        prox_mu = float(getattr(self, "prox_mu", 0.003))
        if prox_mu > 0.0:
            prox_ref_base = [p.detach().clone() for p in self.shared_base.parameters()]
            prox_ref_head = [p.detach().clone() for p in self.mentee_head.parameters()]

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                    x = x[0]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # --- 新的训练逻辑 (Patch内容) ---
                x_weak = x
                x_strong = self._strong_augment_batch(x) if self.enable_strong_aug else x_weak

                rep = self.shared_base(x_weak)
                output = self.mentor_head(rep.detach())
                CE_loss = self.loss(output, y)

#                 rep_g_strong = self.shared_base(x_strong)
#                 output_g_strong = self.mentee_head(rep_g_strong)
#                 output_g_adj = output_g_strong + adj.view(1, -1)
#                 CE_loss_g = self.loss(output_g_adj, y)

#                 rep_g = self.shared_base(x_weak)
#                 output_g = self.mentee_head(rep_g)
                adj_b = adj.view(1, -1)
                if self._is_femnist and self.enable_strong_aug:
                    rep_g = self.shared_base(x_weak)
                    output_g = self.mentee_head(rep_g)
                    CE_loss_g = self.loss(output_g + adj_b, y)
                    rep_g_strong = self.shared_base(x_strong).detach()
                    output_g_strong = self.mentee_head(rep_g_strong)
                    CE_loss_g = CE_loss_g + 0.25 * self.loss(output_g_strong + adj_b, y)
                else:
                    rep_g_strong = self.shared_base(x_strong)
                    output_g_strong = self.mentee_head(rep_g_strong)
                    output_g_adj = output_g_strong + adj_b
                    CE_loss_g = self.loss(output_g_adj, y)
                    rep_g = self.shared_base(x_weak)
                    output_g = self.mentee_head(rep_g)

                den = (CE_loss.detach() + CE_loss_g.detach() + 1e-12).clamp(min=1.0)

                if self.enable_mentor_ema and self.ema_teacher_distill:
                    with torch.no_grad():
                        teacher_rep = self.ema_shared_base(x_weak)
                        teacher_logit = self.ema_mentor_head(teacher_rep)
                else:
                    teacher_rep = rep.detach()
                    teacher_logit = output.detach()

                with torch.no_grad():
                    if self.enable_asd_mask:
                        pt = F.softmax(teacher_logit, dim=1)
                        conf, pred = pt.max(dim=1)
                        if self.asd_conf > 0.0:
                            mask = (pred == y) & (conf >= self.asd_conf)
                        else:
                            mask = (pred == y)
                    else:
                        mask = torch.ones(y.shape[0], device=y.device, dtype=torch.bool)

                if torch.any(mask):
                    T = self.asd_T
                    adj_b = adj.view(1, -1)
                    out_s = output_g_strong[mask] + adj_b
                    out_t = teacher_logit[mask].detach() + adj_b
                    distill_loss = self.KL(F.log_softmax(out_s / T, dim=1), F.softmax(out_t / T, dim=1)) * (T * T) / den

                    rep_s = rep_g[mask]
                    rep_t = teacher_rep[mask].detach()
                    feature_loss = self.MSE(self.W_h(rep_s), rep_t) / den
                else:
                    distill_loss = output_g.sum() * 0.0
                    feature_loss = output_g.sum() * 0.0
                if not self.enable_asd_logits:
                    distill_loss = distill_loss * 0.0
                if not self.enable_asd_feat:
                    feature_loss = feature_loss * 0.0


                loss_mentor = CE_loss
                loss_student = CE_loss_g + self.asd_beta * distill_loss + self.asd_gamma * feature_loss
                if prox_mu > 0.0:
                    prox = 0.0
                    for p, r in zip(self.shared_base.parameters(), prox_ref_base):
                        prox = prox + (p - r).pow(2).sum()
                    for p, r in zip(self.mentee_head.parameters(), prox_ref_head):
                        prox = prox + (p - r).pow(2).sum()
                    loss_student = loss_student + 0.5 * prox_mu * prox


                self.optimizer.zero_grad(set_to_none=True)

                (loss_mentor + loss_student).backward()

                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.shared_base.parameters(),
                        self.mentee_head.parameters(),
                        self.mentor_head.parameters(),
                        self.W_h.parameters(),
                    ),
                    10,
                )
                self.optimizer.step()
                if self.enable_mentor_ema:
                    self._ema_update()
        self.decomposition()

        if self.learning_rate_decay:
            lr_m = self.optimizer.param_groups[2]["lr"]
            self.learning_rate_scheduler.step()
            self.optimizer.param_groups[2]["lr"] = lr_m

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def _recover_param(self, packed, target_shape):
        u, s, vt = packed
        mat = (u * s[np.newaxis, :]) @ vt
        return mat.reshape(target_shape)

    def set_parameters(self, global_param, energy, global_prior=None):
        self.downlink_bytes = self._dict_nbytes(global_param)

        recovered = {}
        for name, old_param in self.global_model.named_parameters():
            if name not in global_param:
                continue
            v = global_param[name]
            if isinstance(v, list) and len(v) == 3:
                recovered[name] = self._recover_param(v, tuple(old_param.data.shape))
            else:
                recovered[name] = v

        for name, old_param in self.global_model.named_parameters():
            if name in recovered:
                old_param.data = torch.tensor(recovered[name], device=self.device).data.clone()

        self.energy = energy
        if global_prior is not None:
            if isinstance(global_prior, torch.Tensor):
                pg = global_prior.detach().cpu().to(torch.float32)
            else:
                pg = torch.tensor(global_prior, dtype=torch.float32)
            pg = torch.clamp(pg, min=1e-6)
            pg = pg / pg.sum()
            self.prior_global = pg
        if self.enable_mentor_ema:
            for ep, p in zip(self.ema_shared_base.parameters(), self.shared_base.parameters()):
                ep.data.copy_(p.data)
            for eb, b in zip(self.ema_shared_base.buffers(), self.shared_base.buffers()):
                eb.data.copy_(b.data)
            for ep, p in zip(self.ema_mentor_head.parameters(), self.mentor_head.parameters()):
                ep.data.copy_(p.data)
            for eb, b in zip(self.ema_mentor_head.buffers(), self.mentor_head.buffers()):
                eb.data.copy_(b.data)


    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        # self._label_debug_dump(trainloader)    # ← 放在这行（新加）
        self.model.train()
        self.global_model.train()
        self.model.eval()
        self.global_model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.shared_base(x)
                rep_g = self.shared_base(x)
                output = self.mentor_head(rep)
                output_g = self.mentee_head(rep_g)

                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                

                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

                loss = CE_loss + L_d + L_h
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
    def _svd_pack_2d_torch(self, mat_t, energy, raw_bytes=None, min_compression_ratio=0.95):
        if not torch.is_tensor(mat_t):
            mat_t = torch.as_tensor(np.asarray(mat_t), dtype=torch.float32)
        else:
            if mat_t.dtype != torch.float32:
                mat_t = mat_t.float()

        m = int(mat_t.shape[0])
        n = int(mat_t.shape[1])
        if raw_bytes is None:
            raw_bytes = int(m * n * 4)

        max_allowed_floats = (raw_bytes / 4.0) * float(min_compression_ratio)
        r_budget = int(max_allowed_floats // (m + n + 1))
        full_rank = int(min(m, n))
        if r_budget >= full_rank:
            return None

        with torch.no_grad():
            u, s, vh = torch.linalg.svd(mat_t, full_matrices=False)
            s2 = s * s
            tot = float(s2.sum().item())
            if tot == 0.0:
                return None
            cum = torch.cumsum(s2, dim=0)
            r_energy = int(torch.searchsorted(cum, energy * tot).item() + 1)
            r_final = max(1, min(r_energy, r_budget, int(s.shape[0])))

            u = u[:, :r_final].contiguous()
            s = s[:r_final].contiguous()
            vh = vh[:r_final, :].contiguous()

        u_np = u.detach().cpu().numpy().astype(np.float32, copy=False)
        s_np = s.detach().cpu().numpy().astype(np.float32, copy=False)
        vt_np = vh.detach().cpu().numpy().astype(np.float32, copy=False)

        comp_bytes = int(u_np.nbytes + s_np.nbytes + vt_np.nbytes)
        if comp_bytes >= int(raw_bytes):
            return None

        return [u_np, s_np, vt_np]

    def decomposition(self):
        self.compressed_param = {}
        device_svd = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        for name, param in self.global_model.named_parameters():
            w_t = param.detach()
            if w_t.dtype != torch.float32:
                w_t = w_t.float()
            w_t = w_t.to(device_svd, non_blocking=True)

            if ("embeddings" in name) or (w_t.ndim <= 1) or (w_t.shape[0] <= 1) or (not torch.is_floating_point(w_t)):
                self.compressed_param[name] = w_t.detach().cpu().numpy().astype(np.float32, copy=False)
                continue

            raw_bytes = int(w_t.numel() * 4)

            if w_t.ndim == 4:
                mat_t = w_t.reshape(w_t.shape[0], -1)
            else:
                mat_t = w_t.reshape(w_t.shape[0], -1)

            packed = self._svd_pack_2d_torch(
                mat_t,
                self.energy,
                raw_bytes=raw_bytes,
                min_compression_ratio=0.95
            )

            if packed is None:
                self.compressed_param[name] = w_t.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                self.compressed_param[name] = packed

        self.uplink_bytes = self._dict_nbytes(self.compressed_param)

    def test_metrics(self):
        self.model.eval()
        self.global_model.eval()
        s_top1, s_n, s_auc, s_top5 = self._test_metrics_with_model(self.global_model)
        m_native_top1, m_native_n, m_native_auc, m_native_top5 = self._test_metrics_with_model(self.model)
        backup_model_state = copy.deepcopy(self.model.state_dict())
        original_grad_states = {}
        for name, param in self.model.named_parameters():
            original_grad_states[name] = param.requires_grad
        for p in self.shared_base.parameters():
            p.requires_grad_(False)
        for p in self.mentor_head.parameters():
            p.requires_grad_(True)
        self.shared_base.eval()
        self.mentor_head.train()
        ft_lr = float(getattr(self, "learning_rate", 0.01))
        ft_epochs = int(getattr(self, "ft_epochs", 1))
        ft_batches = int(getattr(self, "ft_batches", 10))
        ft_optimizer = torch.optim.SGD(self.mentor_head.parameters(), lr=ft_lr)
        ft_loss_fn = nn.CrossEntropyLoss()
        for _ in range(ft_epochs):
            trainloader = self.load_train_data()
            for bi, (x, y) in enumerate(trainloader):
                if bi >= ft_batches:
                    break
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                ft_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    rep = self.shared_base(x)
                out = self.mentor_head(rep)
                loss = ft_loss_fn(out, y)
                loss.backward()
                ft_optimizer.step()
        self.model.eval()
        m_ft_top1, m_ft_n, m_ft_auc, m_ft_top5 = self._test_metrics_with_model(self.model)
        self.model.load_state_dict(backup_model_state)
        for name, param in self.model.named_parameters():
            if name in original_grad_states:
                param.requires_grad = original_grad_states[name]
            else:
                param.requires_grad = True
        self.model.eval()
        self.global_model.eval()
        return {
            "mentor": (m_native_top1, m_native_n, m_native_auc, m_native_top5),
            "mentor_posthoc": (m_ft_top1, m_ft_n, m_ft_auc, m_ft_top5),
            "mentee": (s_top1, s_n, s_auc, s_top5),
        }
    # def test_metrics(self):
    #     backup_model_state = copy.deepcopy(self.model.state_dict())
    #     original_grad_states = {}
    #     for name, param in self.model.named_parameters():
    #         original_grad_states[name] = param.requires_grad
    #     self.model.eval()
    #     self.global_model.eval()
    #     s_top1, s_n, s_auc, s_top5 = self._test_metrics_with_model(self.global_model)
    #     for p in self.shared_base.parameters():
    #         p.requires_grad_(False)
    #     for p in self.mentor_head.parameters():
    #         p.requires_grad_(True)
    #     self.shared_base.eval()
    #     self.mentor_head.train()
    #     ft_lr = float(getattr(self, "learning_rate", 0.01))
    #     ft_epochs = int(getattr(self, "ft_epochs", 1))
    #     ft_batches = int(getattr(self, "ft_batches", 10))
    #     ft_optimizer = torch.optim.SGD(self.mentor_head.parameters(), lr=ft_lr)
    #     ft_loss_fn = nn.CrossEntropyLoss()
    #     for _ in range(ft_epochs):
    #         trainloader = self.load_train_data()
    #         for bi, (x, y) in enumerate(trainloader):
    #             if bi >= ft_batches:
    #                 break
    #             if isinstance(x, list):
    #                 x = x[0]
    #             x, y = x.to(self.device), y.to(self.device)
    #             ft_optimizer.zero_grad(set_to_none=True)
    #             with torch.no_grad():
    #                 rep = self.shared_base(x)
    #             out = self.mentor_head(rep)
    #             loss = ft_loss_fn(out, y)
    #             loss.backward()
    #             ft_optimizer.step()
    #     self.model.eval()
    #     m_top1, m_n, m_auc, m_top5 = self._test_metrics_with_model(self.model)
    #     self.model.load_state_dict(backup_model_state)
    #     for name, param in self.model.named_parameters():
    #         if name in original_grad_states:
    #             param.requires_grad = original_grad_states[name]
    #         else:
    #             param.requires_grad = True
    #     self.model.eval()
    #     self.global_model.eval()
    #     return {"mentor": (m_top1, m_n, m_auc, m_top5), "mentee": (s_top1, s_n, s_auc, s_top5)}

    def _label_debug_dump(self, loader=None):
        if getattr(self, "_label_checked", False):
            return
        try:
            # 允许外部把现成的 trainloader 传进来；否则这里再取一次
            if loader is None:
                loader = self.load_train_data()

            ys = []
            for _, y in itertools.islice(loader, 2):  # 看 1~2 个 batch 就够
                ys.append(y.detach().cpu())
            if not ys:
                print(f"[LabelCheck][client {getattr(self,'id',-1)}] empty trainloader?")
                return
            by = torch.cat(ys)
            u = torch.unique(by)
            u_list = u.tolist()
            print(f"[LabelCheck][client {getattr(self,'id',-1)}] "
                  f"min={int(by.min())} max={int(by.max())} "
                  f"num_unique={len(u_list)} head={u_list[:20]}")

            # 打印前几个类别名（可选）
            try:
                ROOT = "./dataset/Cifar100"
                with open(os.path.join(ROOT, "cifar-100-python", "meta"), "rb") as f:
                    meta = pickle.load(f, encoding="latin1")
                names = meta["fine_label_names"]
                demo_names = [names[int(t)] for t in by[:16]]
                print(f"[LabelCheck][client {getattr(self,'id',-1)}] sample names: {demo_names}")
            except Exception as e:
                print(f"[LabelCheck] names lookup skipped: {e}")

            # 粗筛：如果从 0 连续起步，容易是“本地重标”
            K = min(10, len(u_list))
            if set(u_list[:K]) == set(range(K)):
                print(f"[WARN][client {getattr(self,'id',-1)}] labels look like 0..K-1 contiguous; possible remap!")
        finally:
            self._label_checked = True