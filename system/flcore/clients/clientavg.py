import copy
import torch
import numpy as np
import time
import torch.nn as nn
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

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

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                out = self.model(x)
                loss = self.loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def test_metrics(self):
        mentee = self._test_metrics_with_model(self.model)

        # ft_epochs = int(getattr(self.args, "ft_epochs", 1))
        # ft_batches = int(getattr(self.args, "ft_batches", 10))
        # ft_lr = float(getattr(self.args, "ft_lr", self.learning_rate * 0.1))
        ft_lr = float(getattr(self, "learning_rate", 0.01))
        ft_epochs = int(getattr(self, "ft_epochs", 1))
        ft_batches = int(getattr(self, "ft_batches", 10))
        ft_calib_batches = int(getattr(self.args, "ft_calib_batches", ft_batches))

        if ft_epochs <= 0 or ft_batches <= 0:
            return {"mentor": mentee, "mentee": mentee}

        backup_state = copy.deepcopy(self.model.state_dict())
        original_grad_states = {n: p.requires_grad for n, p in self.model.named_parameters()}
        prev_training = bool(self.model.training)

        self.model.load_state_dict(backup_state, strict=True)

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

       # ft_optimizer = torch.optim.SGD(ft_params, lr=ft_lr, momentum=0.9)
        ft_optimizer = torch.optim.SGD(ft_params, lr=ft_lr)
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

        return {"mentor": mentor, "mentee": mentee}
