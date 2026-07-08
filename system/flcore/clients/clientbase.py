import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.metrics import accuracy_topk 

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
       # torch.manual_seed(0)
        self.args = args
        self.seed = int(getattr(args, "seed", 0))
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self._train_data = None
        self._test_data = None
        self._train_loader = None
        self._test_loader = None
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        # 1. 缓存 Dataset 对象（避免重复读取磁盘）
        if self._train_data is None:
            self._train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        
        # 2. 缓存 DataLoader 对象（避免重复创建开销），并注入 Generator
        if self._train_loader is None or batch_size != self.batch_size:
            # --- 新增 Generator 逻辑 ---
            
            self._train_loader = DataLoader(
                self._train_data, 
                batch_size, 
                drop_last=True, 
                shuffle=True, 
            )
            
        return self._train_loader

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
            
        if self._test_data is None:
            self._test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
            
        if self._test_loader is None or batch_size != self.batch_size:
            self._test_loader = DataLoader(
                self._test_data, 
                batch_size, 
                drop_last=False, 
                shuffle=False, # 注意：测试集一般不需要 Shuffle，除非你想评估 Variance
            )
            
        return self._test_loader
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

#     def test_metrics(self):
#         testloaderfull = self.load_test_data()
#         # self.model = self.load_model('model')
#         # self.model.to(self.device)
#         self.model.eval()

#         test_acc = 0
#         test_num = 0
#         y_prob = []
#         y_true = []
        
#         with torch.no_grad():
#             for x, y in testloaderfull:
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 output = self.model(x)

#                 test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
#                 test_num += y.shape[0]

#                 y_prob.append(output.detach().cpu().numpy())
#                 nc = self.num_classes
#                 if self.num_classes == 2:
#                     nc += 1
#                 lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
#                 if self.num_classes == 2:
#                     lb = lb[:, :2]
#                 y_true.append(lb)

#         # self.model.cpu()
#         # self.save_model(self.model, 'model')

#         y_prob = np.concatenate(y_prob, axis=0)
#         y_true = np.concatenate(y_true, axis=0)

#         auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
#         return test_acc, test_num, auc
    def _test_metrics_with_model(self, model):
        testloaderfull = self.load_test_data()
        model.eval()

        top1_correct = 0
        top5_correct = 0
        test_num = 0

        need_auc = (self.num_classes >= 2)
        if need_auc:
            y_prob, y_true = [], []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)
                t1, t5 = accuracy_topk(logits, y, topk=(1,5))
                b = y.size(0)
                top1_correct += t1 * b
                top5_correct += t5 * b
                test_num += b

#                 if need_auc:
#                     y_prob.append(logits.detach().cpu().numpy())
#                     lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes))
#                     y_true.append(lb)

#         auc = None
#         if need_auc:
#             y_prob = np.concatenate(y_prob, axis=0)
#             y_true = np.concatenate(y_true, axis=0)
#             try:
#                 if self.num_classes == 2:
#                     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
#                 else:
#                     auc = metrics.roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
#             except Exception as e:
#                 auc = None  # AUC不可计算时返回None
                if need_auc:
                        # 1) 用 softmax 做成概率，数值更稳
                        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                        # 2) 二值化标签
                        lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes))
                        y_prob.append(probs)
                        y_true.append(lb)

        auc = None
        if need_auc and len(y_true) > 0:
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            # 3) 过滤非有限值
            if not np.isfinite(y_prob).all():
                y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)

            try:
                # 4) 二分类：只取正类列
                if self.num_classes == 2:
                    pos = 1  # 以类别 1 为正类
                    yt = y_true[:, pos]
                    yp = y_prob[:, pos]
                    if yt.min() == yt.max():  # 只有一个标签
                        auc = None
                    else:
                        auc = metrics.roc_auc_score(yt, yp)

                # 5) 多分类：仅保留“在该客户端出现过的类”
                else:
                    present = (y_true.sum(axis=0) > 0)  # 至少有正样本  
                    if present.sum() < 2:
                        auc = None
                    else:
                        auc = metrics.roc_auc_score(
                            y_true[:, present],
                            y_prob[:, present],
                            average='macro',
                            multi_class='ovr'
                        )
                # 6) 再做一次有限性检查
                if auc is not None and not np.isfinite(auc):
                    auc = None
            except Exception:
                auc = None
        return top1_correct, test_num, auc, top5_correct

    def test_metrics(self):
        return self._test_metrics_with_model(self.model)

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
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
