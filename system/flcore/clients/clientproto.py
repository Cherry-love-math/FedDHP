import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
from utils.metrics import accuracy_topk 
from sklearn.preprocessing import label_binarize
from sklearn import metrics
class clientProto(Client):
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

        protos = defaultdict(list)
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

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        # rep = self.model.base(x)
        # print(torch.sum(rep!=0).item() / rep.numel())

        # self.collect_protos()
        self.protos = agg_func(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_protos(self, global_protos):
        self.global_protos = global_protos

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

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        top1_correct = 0
        top5_correct = 0
        test_num = 0
        
        # 准备计算 AUC 所需的变量
        need_auc = (self.num_classes >= 2)
        if need_auc:
            y_prob, y_true = [], []

        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    rep = self.model.base(x)

                    # 1. 计算每个样本到所有类原型的距离矩阵 [batch_size, num_classes]
                    # 初始化为无穷大
                    dist_matrix = torch.full((y.shape[0], self.num_classes), float('inf')).to(self.device)
                    
                    for i, r in enumerate(rep):
                        for j, pro in self.global_protos.items():
                            # 确保该类原型已存在且不是空列表
                            if not isinstance(pro, list):
                                # 计算 MSE 距离
                                dist_matrix[i, j] = self.loss_mse(r, pro)

                    # 2. 将距离转换为“伪 Logits”
                    # 在原型搜索中，距离越小概率越大，所以取负号
                    logits = -dist_matrix 

                    # 3. 使用基类提供的工具函数计算 Top-1 和 Top-5
                    t1, t5 = accuracy_topk(logits, y, topk=(1, 5))
                    batch_size = y.size(0)
                    top1_correct += t1 * batch_size
                    top5_correct += t5 * batch_size
                    test_num += batch_size

                    # 4. AUC 统计逻辑 (参考基类实现)
                    if need_auc:
                        # 使用 Softmax 将负距离转为概率分布
                        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                        lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes))
                        y_prob.append(probs)
                        y_true.append(lb)

            # 5. 计算 AUC (直接复用基类的多分类/二分类处理逻辑)
            auc = None
            if need_auc and len(y_true) > 0:
                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                try:
                    if self.num_classes == 2:
                        auc = metrics.roc_auc_score(y_true[:, 1], y_prob[:, 1])
                    else:
                        present = (y_true.sum(axis=0) > 0)
                        if present.sum() >= 2:
                            auc = metrics.roc_auc_score(y_true[:, present], y_prob[:, present], 
                                                        average='macro', multi_class='ovr')
                except:
                    auc = None

            return top1_correct, test_num, auc, top5_correct
        else:
            # 如果没有全局原型，返回全 0（避免 Server 崩溃）
            return 0, 1e-5, 0, 0

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
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


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