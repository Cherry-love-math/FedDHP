import numpy as np
import time
import copy
import torch
import torch.nn as nn
from flcore.optimizers.fedoptimizer import pFedMeOptimizer
from flcore.clients.clientbase import Client
class clientpFedMe(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lamda = args.lamda
        self.K = args.K
        self.personalized_learning_rate = args.p_learning_rate
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))
        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)
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
    def test_metrics_ft(self):
        ft_epochs = int(getattr(self.args, "ft_epochs", 1))
        ft_batches = int(getattr(self.args, "ft_batches", 10))
        ft_lr = float(getattr(self.args, "ft_lr", self.learning_rate * 0.1))
        ft_calib_batches = int(getattr(self.args, "ft_calib_batches", ft_batches))
        mentee = self._test_metrics_with_model(self.model)
        if ft_epochs <= 0 or ft_batches <= 0:
            return mentee
        backup_state = copy.deepcopy(self.model.state_dict())
        original_grad_states = {n: p.requires_grad for n, p in self.model.named_parameters()}
        prev_training = bool(self.model.training)
        for p in self.model.parameters():
            p.requires_grad_(False)
        ft_module = None
        for attr in ("head", "fc", "classifier", "linear"):
            m = getattr(self.model, attr, None)
            if isinstance(m, nn.Module):
                ft_module = m
                break
        if ft_module is not None:
            ft_params = list(ft_module.parameters())
            for p in ft_params:
                p.requires_grad_(True)
        else:
            for p in self.model.parameters():
                p.requires_grad_(True)
            ft_params = [p for p in self.model.parameters() if p.requires_grad]
        trainloader = self.load_train_data()
        if ft_calib_batches > 0:
            self._bn_calib(self.model, trainloader, ft_calib_batches)
        self.model.eval()
        ft_optimizer = torch.optim.SGD(ft_params, lr=ft_lr, momentum=0.9)
        ft_loss_fn = self.loss
        for _ in range(ft_epochs):
            for bi, (x, y) in enumerate(trainloader):
                if bi >= ft_batches:
                    break
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                ft_optimizer.zero_grad(set_to_none=True)
                out = self.model(x)
                loss = ft_loss_fn(out, y)
                loss.backward()
                ft_optimizer.step()
        mentor = self._test_metrics_with_model(self.model)
        self.model.load_state_dict(backup_state, strict=True)
        for n, p in self.model.named_parameters():
            p.requires_grad_(original_grad_states.get(n, True))
        if prev_training:
            self.model.train()
        else:
            self.model.eval()
        return mentor
    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                for i in range(self.K):
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step(self.local_params, self.device)
                self.personalized_params = [p.detach().clone() for p in self.model.parameters()]
                with torch.no_grad():
                    for new_param, localweight in zip(self.personalized_params, self.local_params):
                        localweight.data.add_(new_param.data - localweight.data, alpha=self.lamda * self.learning_rate)
                self.update_parameters(self.model, self.local_params)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time
    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        self.update_parameters(self.model, self.personalized_params)
        self.model.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        return test_acc, test_num
    def train_metrics_personalized(self):
        trainloader = self.load_train_data()
        self.update_parameters(self.model, self.personalized_params)
        self.model.eval()
        train_acc = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y).item()
                lm = torch.cat([p.data.view(-1) for p in self.local_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.personalized_params], dim=0)
                loss += 0.5 * self.lamda * torch.norm(lm - pm, p=2).item()
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_num += y.shape[0]
                losses += loss * y.shape[0]
        return train_acc, losses, train_num