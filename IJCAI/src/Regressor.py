""" strongly inspired by Ferdinando Fioretto's code """
import network, dataset, agent

from torch_inputs import *


class Regressor(agent.AbstractAgent):
    def __init__(self, params, d_train, d_test, d_val):
        super(Regressor, self).__init__(params,d_train, d_test, d_val)
        self._train_data = d_train
        net_par = {'i_dim': self._train_data.n_var,
                   'o_dim': 1,
                   'h_dim': 10,
                   'n_layers': 3}
        self._model = network.Net(net_par)
        # Optimizers, for each dataset partition
        self._optimizers = torch.optim.Adam(self._model.parameters(), lr=0.001)
        # Loss functions for each dataset partition
        self._loss = nn.MSELoss()

    def train(self):
        for epoch in range(self._nepochs):
            for (x, y) in self._train_data:
                y_pred = self.predict(x)
                loss = self.compute_loss(y_pred, y, x)
                self.propagate_loss(loss)
            self.validation_step(epoch)

    ''' Override the compute_loss method in AbstractAgent class'''

    def compute_loss(self, y_pred, y, x=None):
        loss = self._loss(y_pred, y)
        return loss

    ''' Override the propoagate_loss method in AbstractAgent class'''

    def propagate_loss(self, loss):
        self._optimizers.zero_grad()
        loss.backward(retain_graph=True)
        self._optimizers.step()

    ''' Override predict method in AbstractAgent class'''

    def predict(self, x):
        # Predict on each data partition
        y_pred = self._model(x)
        return y_pred


