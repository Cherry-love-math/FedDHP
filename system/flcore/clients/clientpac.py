import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict


class clientPAC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda


    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            rep = self.model.base(x)
            output = self.model.head(rep)
            loss = self.loss(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False
            
        # protos = defaultdict(list)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                # for i, yy in enumerate(y):
                #     y_c = yy.item()
                #     protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        # rep = self.model.base(x)
        # print(torch.sum(rep!=0).item() / rep.numel())

        self.collect_protos()
        # self.protos = agg_func(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        self.V, self.h = self.statistics_extraction()

    def set_head(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    # https://github.com/JianXu95/FedPAC/blob/main/methods/fedpac.py#L126
    def statistics_extraction(self):
        model = self.model
        trainloader = self.load_train_data()

        counts = None
        feat_sum = None
        sqnorm_sum = None
        d = None
        feat_dtype = None

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                    xb = x[0]
                else:
                    x = x.to(self.device)
                    xb = x
                y = y.to(self.device).long()

                feat = model.base(xb)

                if d is None:
                    d = feat.shape[1]
                    feat_dtype = feat.dtype
                    counts = torch.zeros(self.num_classes, device=self.device, dtype=feat_dtype)
                    feat_sum = torch.zeros((self.num_classes, d), device=self.device, dtype=feat_dtype)
                    sqnorm_sum = torch.zeros(self.num_classes, device=self.device, dtype=feat_dtype)

                ones = torch.ones_like(y, dtype=feat_dtype)
                counts.index_add_(0, y, ones)
                feat_sum.index_add_(0, y, feat)
                sqnorm_sum.index_add_(0, y, (feat * feat).sum(dim=1))

        if d is None:
            return 0.0, torch.zeros((self.num_classes, 1), device=self.device)

        total_n = counts.sum().clamp_min(1.0)
        py = counts / total_n
        h_ref = feat_sum / total_n

        v = torch.tensor(0.0, device=self.device, dtype=feat_dtype)
        nz = counts > 0
        if nz.any():
            mu = torch.zeros_like(feat_sum)
            mu[nz] = feat_sum[nz] / counts[nz].unsqueeze(1)
            term1 = py[nz] * (sqnorm_sum[nz] / counts[nz])
            term2 = (py[nz] * py[nz]) * (mu[nz] * mu[nz]).sum(dim=1)
            v = (term1 - term2).sum()

        v = float(v.item()) / max(float(self.train_samples), 1.0)

        return v, h_ref

    def _bn_calib(self, model, loader, batches):
        model.train()
        with torch.no_grad():
            for bi, (x, y) in enumerate(loader):
                if bi >= batches:
                    break
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                _ = model(x)
        model.eval()

    def test_metrics(self):
        personalized = self._test_metrics_with_model(self.model)
        return personalized

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos