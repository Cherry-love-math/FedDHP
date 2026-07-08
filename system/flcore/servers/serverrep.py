import random
import time
from flcore.clients.clientrep import clientRep
from flcore.servers.serverbase import Server
from threading import Thread
import json  # [新增]
import os    # [新增]

class FedRep(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRep)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap  # 让 TensorBoard 曲线横轴紧凑 (0, 1, 2...)
                self.evaluate()
                self.current_round = _orig_round         # 恢复真实轮次，供后续通信记录使用
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRep)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        

#     def receive_models(self):
#         assert (len(self.selected_clients) > 0)

#         active_clients = random.sample(
#             self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

#         self.uploaded_weights = []
#         self.uploaded_models = []
#         tot_samples = 0
#         for client in active_clients:
#             client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
#                     client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
#             if client_time_cost <= self.time_threthold:
#                 tot_samples += client.train_samples
#                 self.uploaded_weights.append(client.train_samples)
#                 self.uploaded_models.append(client.model.base)
#         for i, w in enumerate(self.uploaded_weights):
#             self.uploaded_weights[i] = w / tot_samples
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        
        # [新增] 上行流量计数器
        up = 0

        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                
                # FedRep 上传的是 client.model.base
                self.uploaded_models.append(client.model.base)
                
                # [新增] 累加模型大小
                try:
                    up += int(self._model_bytes(client.model.base))
                except:
                    pass

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        # [新增] 记录 Uplink 和 Total
        self.round_uplink_bytes.append(int(up))
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            down = self.round_downlink_bytes[-1] if self.round_downlink_bytes else 0
            self.tb.add_scalar("comm/total_MB", (up + down) / (1024 * 1024), self.current_round)
            # [新增] 重写 send_models 以记录 Downlink Bytes
    def send_models(self):
        assert (len(self.clients) > 0)

        # FedRep 实际上只下发 global_model (它充当 Base)
        # 计算模型大小 (利用 serverbase 里的 _model_bytes)
        try:
            per_client = self._model_bytes(self.global_model)
        except:
            per_client = 0
            
        down = int(per_client * len(self.selected_clients))
        self.round_downlink_bytes.append(down)
        self.total_client_rounds += int(len(self.selected_clients))

        # 写入 TensorBoard
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)

        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)