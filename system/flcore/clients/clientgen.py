import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import numpy as np
import copy
class clientGen(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.feature_dim = rep.shape[1]

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

        self.qualified_labels = []
        self.generative_model = None
        self.localize_feature_extractor = args.localize_feature_extractor
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
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

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
                
                labels = np.random.choice(self.qualified_labels, self.batch_size)
                labels = torch.LongTensor(labels).to(self.device)
                z = self.generative_model(labels)
                loss += self.loss(self.model.head(z), labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
  
        
    def set_parameters(self, model, generative_model):
        if self.localize_feature_extractor:
            for new_param, old_param in zip(model.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

        self.generative_model = generative_model

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
                
                labels = np.random.choice(self.qualified_labels, self.batch_size)
                labels = torch.LongTensor(labels).to(self.device)
                z = self.generative_model(labels)
                loss += self.loss(self.model.head(z), labels)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
#     def test_metrics(self):
#         mentee = self._test_metrics_with_model(self.model)

#         ft_epochs = int(getattr(self.args, "ft_epochs", 1))
#         ft_batches = int(getattr(self.args, "ft_batches", 10))
#         ft_lr = float(getattr(self.args, "ft_lr", self.learning_rate))

#         if ft_epochs <= 0 or ft_batches <= 0:
#             return {"mentor": mentee, "mentee": mentee}

#         backup_state = copy.deepcopy(self.model.state_dict())
#         original_grad_states = {name: p.requires_grad for name, p in self.model.named_parameters()}
#         prev_mode = bool(self.model.training)

#         self.model.eval()

#         for p in self.model.parameters():
#             p.requires_grad_(False)

#         ft_module = None
#         for attr in ("head", "fc", "classifier", "linear"):
#             m = getattr(self.model, attr, None)
#             if isinstance(m, torch.nn.Module):
#                 ft_module = m
#                 break

#         if ft_module is not None:
#             ft_params = list(ft_module.parameters())
#             for p in ft_params:
#                 p.requires_grad_(True)
#         else:
#             for p in self.model.parameters():
#                 p.requires_grad_(True)
#             ft_params = [p for p in self.model.parameters() if p.requires_grad]

#         ft_optimizer = torch.optim.SGD(ft_params, lr=ft_lr)
#         ft_loss_fn = self.loss

#         trainloader = self.load_train_data()
#         for _ in range(ft_epochs):
#             for bi, (x, y) in enumerate(trainloader):
#                 if bi >= ft_batches:
#                     break
#                 if isinstance(x, list):
#                     x = x[0]
#                 x = x.to(self.device)
#                 y = y.to(self.device)

#                 ft_optimizer.zero_grad(set_to_none=True)
#                 out = self.model(x)
#                 loss = ft_loss_fn(out, y)
#                 loss.backward()
#                 ft_optimizer.step()

#         mentor = self._test_metrics_with_model(self.model)

#         self.model.load_state_dict(backup_state)
#         for name, p in self.model.named_parameters():
#             p.requires_grad = original_grad_states.get(name, True)
#         self.model.train(prev_mode)
#         self.model.eval()

#         return {"mentor": mentor, "mentee": mentee}
#     def test_metrics(self):
#         mentee = self._test_metrics_with_model(self.model)

#         ft_epochs = int(getattr(self.args, "ft_epochs", 1))
#         ft_batches = int(getattr(self.args, "ft_batches", 10))
        
#         # 【改动 1】：对齐学习率默认值为全局学习率的 0.1，并新增 ft_calib_batches 参数获取
#         ft_lr = float(getattr(self.args, "ft_lr", self.learning_rate * 0.1))
#         ft_calib_batches = int(getattr(self.args, "ft_calib_batches", ft_batches))

#         if ft_epochs <= 0 or ft_batches <= 0:
#             return {"mentor": mentee, "mentee": mentee}

#         backup_state = copy.deepcopy(self.model.state_dict())
#         original_grad_states = {name: p.requires_grad for name, p in self.model.named_parameters()}
#         prev_mode = bool(self.model.training)

#         # 【改动 2】：去掉了这里过早的 self.model.eval()，将其移到 BN 校准之后

#         for p in self.model.parameters():
#             p.requires_grad_(False)

#         ft_module = None
#         for attr in ("head", "fc", "classifier", "linear"):
#             m = getattr(self.model, attr, None)
#             if isinstance(m, torch.nn.Module): # 统一使用 torch.nn.Module 或 nn.Module 均可
#                 ft_module = m
#                 break

#         if ft_module is not None:
#             ft_params = list(ft_module.parameters())
#             for p in ft_params:
#                 p.requires_grad_(True)
#         else:
#             for p in self.model.parameters():
#                 p.requires_grad_(True)
#             ft_params = [p for p in self.model.parameters() if p.requires_grad]

#         # 【改动 3】：提前加载数据，并补充缺失的 BN (Batch Normalization) 校准逻辑
#         trainloader = self.load_train_data()
#         if ft_calib_batches > 0:
#             self._bn_calib(self.model, trainloader, ft_calib_batches)
            
#         self.model.eval() # 在数据加载和校准后，再设置为 eval 模式

#         # 【改动 4】：为优化器增加 momentum=0.9，与代码一完全对齐
#         ft_optimizer = torch.optim.SGD(ft_params, lr=ft_lr, momentum=0.9)
#         ft_loss_fn = self.loss

#         for _ in range(ft_epochs):
#             for bi, (x, y) in enumerate(trainloader):
#                 if bi >= ft_batches:
#                     break
#                 if isinstance(x, list):
#                     x = x[0]
#                 x = x.to(self.device)
#                 y = y.to(self.device)
#                 ft_optimizer.zero_grad(set_to_none=True)
#                 out = self.model(x)
#                 loss = ft_loss_fn(out, y)
#                 loss.backward()
#                 ft_optimizer.step()

#         mentor = self._test_metrics_with_model(self.model)

#         # 【改动 5】：加上 strict=True，并修复状态恢复 Bug（避免模型死锁在 eval 模式）
#         self.model.load_state_dict(backup_state, strict=True)
#         for name, p in self.model.named_parameters():
#             p.requires_grad_(original_grad_states.get(name, True))
            
#         if prev_mode:
#             self.model.train()
#         else:
#             self.model.eval()

#         return {"mentor": mentor, "mentee": mentee}
    def test_metrics(self):
        mentee = self._test_metrics_with_model(self.model)

        ft_epochs = int(getattr(self.args, "ft_epochs", 1))
        ft_batches = int(getattr(self.args, "ft_batches", 10))
        ft_lr = float(getattr(self.args, "ft_lr", self.learning_rate))
        ft_calib_batches = int(getattr(self.args, "ft_calib_batches", 0))

        if ft_epochs <= 0 or ft_batches <= 0:
            return {"mentor": mentee, "mentee": mentee}

        backup_state = copy.deepcopy(self.model.state_dict())
        original_grad_states = {name: p.requires_grad for name, p in self.model.named_parameters()}
        prev_mode = bool(self.model.training)

        for p in self.model.parameters():
            p.requires_grad_(False)

        ft_module = None
        for attr in ("head", "fc", "classifier", "linear"):
            m = getattr(self.model, attr, None)
            if isinstance(m, torch.nn.Module):
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
        if ft_module is not None:
            ft_module.train()

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
        for name, p in self.model.named_parameters():
            p.requires_grad_(original_grad_states.get(name, True))

        if prev_mode:
            self.model.train()
        else:
            self.model.eval()

        return {"mentor": mentor, "mentee": mentee}