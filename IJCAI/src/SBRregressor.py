""" strongly inspired by Ferdinando Fioretto's code """
import network, dataset, agent, util

import numpy as np

from torch_inputs import *


class SBRregressor(agent.AbstractAgent):
    def __init__(self, params, d_train, d_test, d_val):
        super(SBRregressor, self).__init__(params, d_train, d_test, d_val)

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
        self._const_avg_batch = 0
        self.count_const()
        self._LR_rate = 1e-5
        self._LR_multiplier = 0
        self._violations_epoch = []

    def count_const(self):
        const_batch = []
        for (x, y) in self._train_data:
            n = max(1, len(util.couples(x.tolist())))
            const_batch.append(n)
        self.const_avg_batch = int(np.mean(np.array(const_batch)))

    # @property
    def const_avg_batch(self):
        return self.const_avg_batch

    def train(self):
        for epoch in range(self._nepochs):
            violations_epoch = []
            for (x, y) in self._train_data:
                M = self.build_kwb_matrix(x)
                y_pred = self.predict(x)
                loss, violation = self.compute_loss(M, y_pred, y, x)
                self.propagate_loss(loss)
                violations_epoch.append(violation)
            self.update_LR_multipliers(violations_epoch)
            self.validation_step(epoch)
            self.print_report(epoch)

    ''' Override the compute_loss method in AbstractAgent class'''

    def compute_loss(self, M, y_pred, y, x=None):
        loss = self._loss(y_pred, y)
        # g(x') - g(x'') where g(x'') is dominant over g(x'), hence should be have grater value,
        # (N.B. we are considering the -log10 of the error otherwise it would be g(x'') - g(x')
        rules = torch.mm(torch.transpose(M, 0, 1), y_pred)
        # filter only the positive values (violations), therefore the violations to the constraint
        v = torch.sum(torch.max(Ten(np.zeros(rules.size()[0])), torch.transpose(rules, 0, 1)))
        loss += self._LR_multiplier * v

        return loss, v

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

    ''' Update the Lagrangian Multipliers associated to the constraint violations'''

    def update_LR_multipliers(self, violations):
        self._LR_multiplier = self._LR_multiplier + (self._LR_rate * torch.sum(Ten(violations)))

    def print_report(self, epoch):
        print('\t LR mult:', self._LR_multiplier)
