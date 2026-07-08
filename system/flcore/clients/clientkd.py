import copy, os, pickle, time, itertools
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from flcore.clients.clientbase import Client

class clientKD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.mentee_learning_rate = args.mentee_learning_rate
        self.model = copy.deepcopy(args.teacher_model).to(self.device)
        self.global_model = copy.deepcopy(args.model).to(self.device)

        self.has_BatchNorm = any(isinstance(layer, nn.BatchNorm2d) for layer in self.model.children())

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.learning_rate_decay_gamma)

        self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.mentee_learning_rate)
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=args.learning_rate_decay_gamma)

        self.feature_dim_t = list(self.model.head.parameters())[0].shape[1]
        self.feature_dim_s = list(self.global_model.head.parameters())[0].shape[1]
        self.W_h = nn.Linear(self.feature_dim_s, self.feature_dim_t, bias=False).to(self.device)
        self.optimizer_W = torch.optim.SGD(self.W_h.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_W, gamma=args.learning_rate_decay_gamma)

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()
        self.compressed_param = {}
        self.energy = None

    def _obj_nbytes(self, obj):
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            return int(np.asarray(obj[0]).nbytes + np.asarray(obj[1]).nbytes + np.asarray(obj[2]).nbytes)
        return int(np.asarray(obj).nbytes)

    def _dict_nbytes(self, d):
        return int(sum(self._obj_nbytes(v) for v in d.values()))

    def train(self):
        trainloader = self.load_train_data()
        self.model.train(); self.global_model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]): x[0] = x[0].to(self.device)
                else: x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow: time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x); rep_g = self.global_model.base(x)
                output = self.model.head(rep); output_g = self.global_model.head(rep_g)

                CE_loss = self.loss(output, y); CE_loss_g = self.loss(output_g, y)
                den = CE_loss.detach() + CE_loss_g.detach() + 1e-12

                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g.detach(), dim=1)) / den
                L_d_g = self.KL(F.log_softmax(output_g, dim=1), F.softmax(output.detach(), dim=1)) / den
                L_h = self.MSE(rep, self.W_h(rep_g.detach())) / den
                L_h_g = self.MSE(rep.detach(), self.W_h(rep_g)) / den

                loss = CE_loss + L_d + L_h
                loss_g = CE_loss_g + L_d_g + L_h_g

                self.optimizer.zero_grad(); self.optimizer_g.zero_grad(); self.optimizer_W.zero_grad()
                loss.backward(retain_graph=True); loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.W_h.parameters(), 10)
                self.optimizer.step(); self.optimizer_g.step(); self.optimizer_W.step()

        self.decomposition()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()
            self.learning_rate_scheduler_W.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _recover_param(self, packed, target_shape):
        u, s, vt = packed
        mat = (u * s[np.newaxis, :]) @ vt
        return mat.reshape(target_shape)

    def set_parameters(self, global_param, energy):
        self.downlink_bytes = self._dict_nbytes(global_param)
        recovered = {}
        for name, old_param in self.global_model.named_parameters():
            if name not in global_param: continue
            v = global_param[name]
            if isinstance(v, list) and len(v) == 3:
                recovered[name] = self._recover_param(v, tuple(old_param.data.shape))
            else:
                recovered[name] = v
        for name, old_param in self.global_model.named_parameters():
            if name in recovered:
                old_param.data = torch.tensor(recovered[name], device=self.device).data.clone()
        self.energy = energy

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval(); self.global_model.eval()
        train_num = 0; losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]): x[0] = x[0].to(self.device)
                else: x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x); rep_g = self.global_model.base(x)
                output = self.model.head(rep); output_g = self.global_model.head(rep_g)

                CE_loss = self.loss(output, y); CE_loss_g = self.loss(output_g, y)
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

                loss = CE_loss + L_d + L_h
                train_num += y.shape[0]; losses += loss.item() * y.shape[0]
        return losses, train_num

    def _svd_pack_2d_torch(self, mat_t, energy, raw_bytes):
        if not torch.is_tensor(mat_t):
            mat_t = torch.as_tensor(mat_t, dtype=torch.float32)
        elif mat_t.dtype != torch.float32:
            mat_t = mat_t.float()

        # 确保数据在 GPU 上 (Client端通常 self.device 就是 GPU)
        if str(self.device) != 'cpu':
             mat_t = mat_t.to(self.device)

        with torch.no_grad():
            try:
                u, s, vh = torch.linalg.svd(mat_t, full_matrices=False)
            except RuntimeError:
                return None
            
            s2 = s * s
            tot = float(s2.sum().item())
            if tot == 0.0: return None
            
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

        return [
            u.detach().cpu().numpy(),
            s.detach().cpu().numpy(),
            vh.detach().cpu().numpy()
        ]

    # 修改 decomposition 使用 torch 版本
    def decomposition(self):
        self.compressed_param = {}
        # 统一使用 self.device，通常是 cuda
        for name, param in self.global_model.named_parameters():
            # 1. 保持在 GPU 上处理
            w_t = param.detach()
            if 'embeddings' in name or param.ndim <= 1:
                self.compressed_param[name] = w_t.cpu().numpy().astype(np.float32, copy=False)
                continue

            w_t = w_t.float()
            # 2. 维度重塑
            mat_t = w_t.reshape(w_t.shape[0], -1)
            raw_bytes = w_t.numel() * 4
            
            # 3. 调用 GPU SVD
            packed = self._svd_pack_2d_torch(mat_t, self.energy, raw_bytes)
            
            if packed is not None:
                self.compressed_param[name] = packed
            else:
                self.compressed_param[name] = w_t.cpu().numpy().astype(np.float32, copy=False)
                
        self.uplink_bytes = self._dict_nbytes(self.compressed_param)
    # def test_metrics(self):
        # m_top1, m_n, m_auc, m_top5 = self._test_metrics_with_model(self.model)
        # s_top1, s_n, s_auc, s_top5 = self._test_metrics_with_model(self.global_model)
        # return {"mentor": (m_top1, m_n, m_auc, m_top5), "mentee": (s_top1, s_n, s_auc, s_top5)}
    def test_metrics(self):
        self.model.eval()
        self.global_model.eval()

        s_top1, s_n, s_auc, s_top5 = self._test_metrics_with_model(self.global_model)
        m_top1, m_n, m_auc, m_top5 = self._test_metrics_with_model(self.model)

        backup_model_state = copy.deepcopy(self.model.state_dict())
        original_grad_states = {}

        for name, param in self.model.named_parameters():
            original_grad_states[name] = param.requires_grad

        for p in self.model.base.parameters():
            p.requires_grad_(False)

        for p in self.model.head.parameters():
            p.requires_grad_(True)

        self.model.base.eval()
        self.model.head.train()

        # ft_lr = float(getattr(self, "learning_rate", 0.01))
        # ft_epochs = int(getattr(self.args, "ft_epochs", 1))
        # ft_batches = int(getattr(self.args, "ft_batches", 10))
        ft_lr = float(getattr(self.args, "ft_lr", self.learning_rate))
        ft_epochs = int(getattr(self.args, "ft_epochs", 1))
        ft_batches = int(getattr(self.args, "ft_batches", 10))
        ft_optimizer = torch.optim.SGD(self.model.head.parameters(), lr=ft_lr)
        ft_loss_fn = nn.CrossEntropyLoss()

        trainloader = self.load_train_data()

        for _ in range(ft_epochs):
            for bi, (x, y) in enumerate(trainloader):
                if bi >= ft_batches:
                    break
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                ft_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    rep = self.model.base(x)
                out = self.model.head(rep)
                loss = ft_loss_fn(out, y)
                loss.backward()
                ft_optimizer.step()

        self.model.eval()
        m_ft_top1, m_ft_n, m_ft_auc, m_ft_top5 = self._test_metrics_with_model(self.model)

        self.model.load_state_dict(backup_model_state, strict=True)

        for name, param in self.model.named_parameters():
            param.requires_grad_(original_grad_states.get(name, True))

        self.model.eval()
        self.global_model.eval()

        return {
            "mentor": (m_top1, m_n, m_auc, m_top5),
            "mentor_posthoc": (m_ft_top1, m_ft_n, m_ft_auc, m_ft_top5),
            "mentee": (s_top1, s_n, s_auc, s_top5),
        }