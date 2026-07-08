import time
import numpy as np
from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from collections import defaultdict
class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientProto)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = None
   #     self.eval_mode = getattr(args, "proto_eval_mode", "head")

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_protos()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
         #       self.set_clients_eval_mode(self.eval_mode)
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    # def set_clients_eval_mode(self, mode):
    #     for c in self.clients:
    #         c.set_eval_mode(mode)

    def send_protos(self):
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        if local_protos is None:
            continue
        for label, proto in local_protos.items():
            agg_protos[label].append(proto)

    for label, proto_list in agg_protos.items():
        if len(proto_list) == 0:
            continue
        if len(proto_list) == 1:
            agg_protos[label] = proto_list[0]
        else:
            proto = 0 * proto_list[0].data
            for p in proto_list:
                proto += p.data
            agg_protos[label] = proto / len(proto_list)

    return dict(agg_protos)
