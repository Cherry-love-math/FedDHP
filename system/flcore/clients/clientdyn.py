import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import torch.nn as nn

class clientDyn(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.alpha = args.alpha

        self.global_model_vector = None
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                if self.global_model_vector is not None:
                    v1 = model_parameter_vector(self.model)
                    d = v1 - self.global_model_vector
                    loss += self.alpha * 0.5 * torch.sum(d * d)
                    loss -= torch.dot(v1, self.old_grad)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.global_model_vector is not None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model_vector = model_parameter_vector(model).detach().clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

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
                loss = self.loss(output, y)

                if self.global_model_vector is not None:
                    v1 = model_parameter_vector(self.model)
                    d = v1 - self.global_model_vector
                    loss += self.alpha * 0.5 * torch.sum(d * d)
                    loss -= torch.dot(v1, self.old_grad)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    def test_metrics(self):
        mentee = self._test_metrics_with_model(self.model)
        ft_epochs = int(getattr(self.args, "ft_epochs", 1))
        ft_batches = int(getattr(self.args, "ft_batches", 10))
        ft_lr = float(getattr(self.args, "ft_lr", self.learning_rate))
        if ft_epochs <= 0 or ft_batches <= 0:
            return {"mentor": mentee, "mentee": mentee}
        backup_state = copy.deepcopy(self.model.state_dict())
        original_grad_states = {name: p.requires_grad for name, p in self.model.named_parameters()}
        prev_mode = bool(self.model.training)
        self.model.eval()
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
        ft_optimizer = torch.optim.SGD(ft_params, lr=ft_lr)
        ft_loss_fn = self.loss
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
                out = self.model(x)
                loss = ft_loss_fn(out, y)
                loss.backward()
                ft_optimizer.step()
        mentor = self._test_metrics_with_model(self.model)
        self.model.load_state_dict(backup_state)
        for name, p in self.model.named_parameters():
            p.requires_grad = original_grad_states.get(name, True)
        if prev_mode:
            self.model.train()
        else:
            self.model.eval()
        return {"mentor": mentor, "mentee": mentee}

def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)