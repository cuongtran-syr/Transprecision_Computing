""" Strongly inspired by Ferdinando Fioretto's code """

import util

from torch_inputs import *
import numpy as np
from dataset import Dataset

class AbstractAgent():
    def __init__(self, params, d_train, d_test, d_val):
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        self._train_data = d_train
        self._test_data = d_test
        self._valid_data = d_val

        self._nepochs = params['epochs']
        self._batchsize = params['batch_size']

        self._model = None
        self._optimizer = None
        self._loss = None

    def train(self):
        for epoch in range(self._nepochs):
            for (x, y) in self._train_data:
                y_pred = self.predict(x)
                loss = self.compute_loss(y_pred, y, x)
                self.propagate_loss(loss)
            self.print_report(epoch)
            self.validation_step(epoch)

    def predict(self, x):
        return self._model(x)

    def propagate_loss(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def validation_step(self, epoch):
        mae = list()
        violated_const_model = 0
        violated_const_dataset = 0
        (X, y) = self._valid_data._dataset
        y_pred = self._model.predict(Ten(X))
        mae.append(util.mae(y, y_pred))
        violated_const_dataset += len(util.violated_const(X, y))
        violated_const_model += len(util.violated_const(X, y_pred))
        print(f"epoch: {epoch}, "
              f"MAE: {np.mean(mae)}, "
              f"Violated constraints dataset: {violated_const_dataset}, "
              f"Violated constraints model: {violated_const_model}")

    def print_report(self, epoch):
        pass

    def test(self):
        mae = list()
        violated_const_dataset = 0
        violated_const_model = 0
        (X, y) = self._test_data._dataset
        y_pred = self._model.predict(Ten(X))
        mae = util.mae(y, y_pred)
        violated_const_dataset += len(util.violated_const(X, y))
        violated_const_model += len(util.violated_const(X, y_pred))
        print(f"Test precision: {mae}, "
              f"Violated constraints Dataset: {violated_const_dataset}, "
              f"Violated constraints Model: {violated_const_model}, "
              f"Duplicates: {util.duplicates(y_pred)}")
        return (mae, violated_const_model)


    def build_kwb_matrix(self, data):
        """ Build matrix containg dominance informatins

        Every couple (dominant, dominated) is tracked in the matrix.
        Each row represents a couple, while each column a training sample.
        Dominand samples are marked with -1, while the dominated with 1.
        """
        couples = util.couples(data.tolist())
        n = max(1, len(couples))
        kwb_matrix = np.zeros((n, len(data)))
        for (k, (i, j)) in enumerate(couples):
            kwb_matrix[k, i] = -1
            kwb_matrix[k, j] = 1
        return Ten(kwb_matrix.T)

    def compute_loss(self, y_pred, y, x=None):
        pass

    def plot(self):
        pass
