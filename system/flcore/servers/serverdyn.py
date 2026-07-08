import copy
import time
import random
import torch
from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
from threading import Thread


class FedDyn(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.round_uplink_bytes = []
        self.round_downlink_bytes = []
        self.total_client_rounds = 0

        # Select slow clients
        self.set_slow_clients()
        self.set_clients(clientDyn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.alpha = args.alpha
        
        # Initialize server state (same structure as global model, zero-initialized)
        self.server_state = copy.deepcopy(args.model)
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)

    def _model_bytes(self, model):
        """Estimate the size (in bytes) of a model's state dict."""
        try:
            sd = model.state_dict()
        except Exception:
            sd = {}
        total = 0
        for v in sd.values():
            if torch.is_tensor(v):
                total += v.nelement() * v.element_size()
        return int(total)

    def send_models(self):
        """Send global model to selected clients (downlink)."""
        assert len(self.selected_clients) > 0
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        """Receive models from active (non-dropped) clients (uplink)."""
        assert len(self.selected_clients) > 0

        # Simulate client dropout
        num_active = int((1 - self.client_drop_rate) * self.current_num_join_clients)
        active_clients = random.sample(self.selected_clients, num_active)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        up = 0

        for client in active_clients:
            # Estimate client time cost
            try:
                client_time_cost = (
                    client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] +
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
                )
            except ZeroDivisionError:
                client_time_cost = 0

            # Only accept models from clients within time threshold
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

                try:
                    up += self._model_bytes(client.model)
                except Exception:
                    pass  # Silently ignore byte calculation errors

        # Normalize weights by total samples
        if tot_samples > 0:
            self.uploaded_weights = [w / tot_samples for w in self.uploaded_weights]

        self.round_uplink_bytes.append(int(up))

        # Log to TensorBoard if available
        if hasattr(self, "tb") and self.tb is not None:
            self.tb.add_scalar("comm/uplink_MB", up / (1024 * 1024), self.current_round)
            down = self.round_downlink_bytes[-1] if self.round_downlink_bytes else 0
            self.tb.add_scalar("comm/total_MB", (up + down) / (1024 * 1024), self.current_round)

    def train(self):
        """Main training loop for FedDyn."""
        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()

            # Client selection
            self.selected_clients = self.select_clients()

            # Downlink communication: send global model
            try:
                per_client = self._model_bytes(self.global_model)
            except Exception:
                per_client = 0
            down = int(per_client * len(self.selected_clients))
            self.round_downlink_bytes.append(down)
            self.total_client_rounds += len(self.selected_clients)

            if hasattr(self, "tb") and self.tb is not None:
                self.tb.add_scalar("comm/downlink_MB", down / (1024 * 1024), self.current_round)

            self.send_models()

            # Periodic evaluation
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                _orig_round = self.current_round
                self.current_round = i // self.eval_gap
                self.evaluate()
                self.current_round = _orig_round

            # Train selected clients (sequentially; threading commented out)
            for client in self.selected_clients:
                client.train()

            # Alternative parallel version (currently disabled):
            # threads = [Thread(target=client.train) for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            # Receive models from clients
            self.receive_models()

            # Optional: Deep Leakage from Gradients (DLG) evaluation
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # FedDyn-specific updates
            self.update_server_state()
            self.aggregate_parameters()

            # Record time budget
            elapsed = time.time() - s_t
            self.Budget.append(elapsed)
            print('-' * 50, self.Budget[-1])

            # Early stopping check
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Final results
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAveraged time per iteration.")
        if len(self.Budget) > 1:
            avg_time = sum(self.Budget[1:]) / len(self.Budget[1:])
            print(avg_time)
        else:
            print("Not enough rounds to compute average time.")

        self.save_results()
        self.save_global_model()

        # Evaluate new clients if any
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDyn)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

        # Clean up TensorBoard writer
        if hasattr(self, "tb") and self.tb is not None:
            try:
                self.tb.flush()
            except Exception:
                pass
            self.tb.close()
            self.tb = None

    def add_parameters(self, client_model):
        """Add weighted client model to global model (uniform weight here)."""
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / self.num_join_clients

    def aggregate_parameters(self):
        """Aggregate uploaded models with FedDyn correction using server state."""
        assert len(self.uploaded_models) > 0

        # Initialize global model as zero
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        # Sum all client models
        for client_model in self.uploaded_models:
            self.add_parameters(client_model)

        # Apply FedDyn correction: subtract (1/α) * server_state
        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1.0 / self.alpha) * state_param.data

    def update_server_state(self):
        """Update the server state according to FedDyn algorithm."""
        assert len(self.uploaded_models) > 0

        # Compute average model delta: (client_model - global_model) averaged over all clients
        model_delta = copy.deepcopy(self.uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        # Accumulate (client - global) for each client
        for client_model in self.uploaded_models:
            for server_param, client_param, delta_param in zip(
                self.global_model.parameters(),
                client_model.parameters(),
                model_delta.parameters()
            ):
                delta_param.data += (client_param.data - server_param.data) / self.num_clients

        # Update server state: state -= α * model_delta
        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param.data