import torch
import torch.optim as optim
import copy

class OptimizerWrapper:
    def __init__(self, model, lr: float=1e-3, optimizer_name: str = 'adam'):
        self.model = model
        self.lr = lr
        self.optimizer = self._get_optimizer(optimizer_name)

    def _get_optimizer(self, optimizer_name: str):
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def step(self, grads):
        self.optimizer.step()
        return self.model

    def get_params(self):
        return self.model

class ValueOptimizerWrapper(OptimizerWrapper):
    def __init__(self, model, lr: float=1e-3, optimizer_name: str = 'adam'):
        super().__init__(model, lr, optimizer_name)
        self.target_q_network = copy.deepcopy(model)

    def update_target_params(self):
        self.target_q_network.load_state_dict(self.model.state_dict())

    def get_target_params(self):
        return self.target_q_network
