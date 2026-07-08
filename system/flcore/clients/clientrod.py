import copy
import torch
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.metrics import accuracy_topk 

class clientROD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
                
        self.head = copy.deepcopy(self.model.head)
        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.opt_head, 
            gamma=args.learning_rate_decay_gamma
        )

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1


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
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.optimizer.zero_grad()
                loss_bsm.backward()
                self.optimizer.step()

                out_p = self.head(rep.detach())
                loss = self.loss(out_g.detach() + out_p, y)
                self.opt_head.zero_grad()
                loss.backward()
                self.opt_head.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_head.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

#     def test_metrics(self, model=None):
#         testloader = self.load_test_data()
#         if model == None:
#             model = self.model
#         model.eval()
#         test_acc = 0
#         test_num = 0
#         y_prob = []
#         y_true = []
        
#         with torch.no_grad():
#             for x, y in testloader:
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 rep = self.model.base(x)
#                 out_g = self.model.head(rep)
#                 out_p = self.head(rep.detach())
#                 output = out_g.detach() + out_p

#                 test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
#                 test_num += y.shape[0]

#                 y_prob.append(F.softmax(output).detach().cpu().numpy())
#                 nc = self.num_classes
#                 if self.num_classes == 2:
#                     nc += 1
#                 lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
#                 if self.num_classes == 2:
#                     lb = lb[:, :2]
#                 y_true.append(lb)

#         y_prob = np.concatenate(y_prob, axis=0)
#         y_true = np.concatenate(y_true, axis=0)

#         auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
#         return test_acc, test_num, auc
    def _test_metrics_native(self):
        testloader = self.load_test_data()
        self.model.eval()
        self.head.eval()

        top1_correct = 0.0
        top5_correct = 0.0
        test_num = 0.0

        need_auc = (self.num_classes >= 2)
        if need_auc:
            y_prob = []
            y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                logits = out_g.detach() + out_p

                t1, t5 = accuracy_topk(logits, y, topk=(1, 5))
                b = y.size(0)
                top1_correct += float(t1) * b
                top5_correct += float(t5) * b
                test_num += b

                if need_auc:
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes))
                    y_prob.append(probs)
                    y_true.append(lb)

        auc = None
        if need_auc and test_num > 0:
            try:
                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)

                if self.num_classes == 2:
                    yt = y_true[:, 1]
                    yp = y_prob[:, 1]
                    if yt.min() != yt.max():
                        auc = metrics.roc_auc_score(yt, yp)
                else:
                    present = (y_true.sum(axis=0) > 0)
                    if present.sum() >= 2:
                        auc = metrics.roc_auc_score(
                            y_true[:, present],
                            y_prob[:, present],
                            average="macro",
                            multi_class="ovr"
                        )
            except Exception:
                auc = None

        return top1_correct, test_num, auc, top5_correct
    def test_metrics(self):
        native = self._test_metrics_native()
        posthoc = self.test_metrics_posthoc()
        return {
            "mentee": native,
            "mentor": native,
            "mentor_posthoc": posthoc,
        }

    def test_metrics_posthoc(self):
        backup_head = copy.deepcopy(self.head.state_dict())
        grad_states_model = {}
        grad_states_head = {}

        for name, param in self.model.named_parameters():
            grad_states_model[name] = param.requires_grad
        for name, param in self.head.named_parameters():
            grad_states_head[name] = param.requires_grad

        for p in self.model.base.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

        self.model.eval()
        self.head.train()

        ft_epochs = int(getattr(self, "ft_epochs", 1))
        ft_batches = int(getattr(self, "ft_batches", 10))
        ft_lr = float(getattr(self, "learning_rate", 0.01))
        ft_optimizer = torch.optim.SGD(self.head.parameters(), lr=ft_lr)
        ft_loss_fn = torch.nn.CrossEntropyLoss()

        trainloader = self.load_train_data()
        for _ in range(ft_epochs):
            for bi, (x, y) in enumerate(trainloader):
                if bi >= ft_batches:
                    break
                if type(x) == type([]):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    rep = self.model.base(x)
                    out_g = self.model.head(rep)

                out_p = self.head(rep.detach())
                logits = out_g.detach() + out_p
                loss = ft_loss_fn(logits, y)

                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()

        posthoc = self._test_metrics_native()

        self.head.load_state_dict(backup_head)
        for name, param in self.model.named_parameters():
            param.requires_grad = grad_states_model.get(name, True)
        for name, param in self.head.named_parameters():
            param.requires_grad = grad_states_head.get(name, True)

        self.model.eval()
        self.head.eval()

        return posthoc
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
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
