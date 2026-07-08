import time
import random
from flcore.clients.clientprox import clientProx
from flcore.servers.serverbase import Server
from threading import Thread


class FedProx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProx)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        up = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                try:
                    up += int(self._model_bytes(client.model))
                except Exception:
                    pass
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        self.round_uplink_bytes.append(int(up))
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            down = self.round_downlink_bytes[-1] if self.round_downlink_bytes else 0
            self.tb.add_scalar(
                "comm/total_MB",
                (up + down) / (1024 * 1024),
                self.current_round,
            )


    def train(self):
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # communication: downlink (full global model to selected clients)
            try:
                per_client = self._model_bytes(self.global_model)
            except Exception:
                per_client = 0
            down = int(per_client * len(self.selected_clients))
            self.round_downlink_bytes.append(down)
            self.total_client_rounds += int(len(self.selected_clients))
            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap
                self.evaluate()
                self.current_round = _orig_round

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

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientProx)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        if hasattr(self, "tb") and self.tb is not None:
            try:
                self.tb.flush()
            except Exception:
                pass
            self.tb.close()
            self.tb = None
