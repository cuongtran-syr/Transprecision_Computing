import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error as sk_mae, mean_squared_error as mse
import  sys, copy, pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
if True:
    path_to_repo = '/content/drive/My Drive/Transprecision_Computing/IJCAI/'
    from google.colab import drive
    drive.mount('/content/drive')
sys.path.insert(1,'/content/drive/My Drive/Transprecision_Computing/IJCAI/')

import torch
from torch import nn
Ten = torch.FloatTensor
iTen = torch.LongTensor

def mae(y_true, y_pred):
    """ Compute Mean Absolute Error

    This function computes MAE on the non log
    error

    Parameters
    ----------
        y_true : list(float)
            True value for
            a given sample of data
        y_pred : list(float)
            Predicted value for
            a given sample of data

    Returns
    -------
        MAE : float
            Mean Absolute Error

    """
    y_pred = np.array([10 ** -y for y in y_pred])
    y_true = np.array([10 ** -y for y in y_true])
    return np.mean(np.abs(y_pred - y_true))


def is_dominant(x, y):
    """ Checks if the configuration x is dominant over y

    Parameters
    ----------
        x : list(float)
            configuration
        y : list(float)
            configuration

    Returns
    -------
        Dominance Truth Value : bool
            True if x is dominant over y, False otherwise

    """
    n = len(x) if isinstance(x, list) else x.shape[0]
    return all([x[i] > y[i] for i in range(n)])


def couples(precision):
    """ Counts number of couples dominant dominated """
    n = len(precision)
    couples = []
    for i in range(n):
        x = np.repeat([precision[i]], n, axis=0)
        dominated_idx = np.where(np.all(x > precision, axis=1))[0]
        couples += [(i, j) for j in list(dominated_idx)]

    return couples


def violated_const(precision, error):
    """ Counts number of violated_const

        if x' is dominant on x'' -> -log10(e(x')) > -log10(e(x''))

    """
    n = len(precision)
    violated_const = [(i, j) for (i, j) in couples(precision) if error[i] < error[j]]

    return violated_const


def duplicates(error):
    """ Computes the number of duplicates in the error predicted, especially,
        sums the number of repetition of the 3 most frequent elements.

        This function is used to check the validity of the results predicted
        by the model. As observed previous experimens, high values in the
        multiplier lead to trivial prediction, i.e. for every instance the prediction
        has often the same outcome
    """
    u, c = np.unique(np.round(error, 5), return_counts=True)
    dup = list(zip(u, c))
    dup.sort(key=lambda x: x[1])
    return sum([dup[-1][1], dup[-2][1], dup[-3][1]]) / len(error)


# this procedure is used to create dataset which have a high ratio of constraint violations.
# this allows for consistent value of the regularizator at training time.
def build_dataset(benchmark, n_data, violations_ratio, seed):
    """ Builds a dataset with the desired amount of violated constraints """
    np.random.seed(seed)
    nerr = 30
    suff_label_target = 'err_ds_'
    n_violated_const = int(n_data * violations_ratio)
    labels_target = [suff_label_target + str(i) for i in range(nerr)]

    # reading dataset from csv
    data_file = 'exp_results_{}.csv'.format(benchmark)
    df = pd.read_csv(path_to_repo + 'data/' + data_file, sep=';')
    n_var = len(list(df.filter(regex='var_*')))  # number of variable in the configuration
    # error is capped at 0.95
    for _label_target in labels_target:
        df[_label_target] = [0.95 if x > 0.95 else x for x in df[_label_target]]
        df[_label_target] = [sys.float_info.min if 0 == x else -np.log10(x) for x in df[_label_target]]
    # preprocessing
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(df.iloc[:, 0:n_var])
    y = np.mean(scaler.fit_transform(df[labels_target]), axis=1, dtype='float32').reshape((-1, 1))
    # inject constraint violations
    dataset_violated_const = violated_const(X, y)
    if len(dataset_violated_const) < n_violated_const:
        # the number of violated constraints we want to inject in the dataset exceed the number
        # of violated constraints available in the dataset
        raise ValueError('The desired number of injected constrait violations is not available')
    idx = set()  # set containing indexes of samples in the final dataset used for the training
    idx_control = set()  # set used to count the number of violated constraints used
    while len(idx) != n_violated_const and len(idx_control) < len(dataset_violated_const):
        k = np.random.randint(len(dataset_violated_const))
        idx_control.add(k)
        (i, j) = dataset_violated_const[k]
        idx.add(i)
        idx.add(j)
    while len(idx) < n_data:
        idx.add(np.random.randint(len(X)))
    idx = list(idx)
    return X[idx], y[idx]




class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        idim = params['i_dim']
        odim = params['o_dim']
        hdim = params['h_dim']
        self._nlayers = params['n_layers']
        self._af = nn.ReLU
        self._of = nn.Linear

        self.i_layer = nn.Sequential(
            nn.Linear(idim, hdim),
            self._af(inplace=True))

        layers = []
        for i in range(self._nlayers-1):
            layers.append(nn.Linear(hdim, hdim))
            layers.append(self._af(inplace=True))
        self.h_layers = nn.Sequential(*layers)

        self.o_layer = nn.Sequential(
            nn.Linear(hdim, odim))

    def forward(self, x):
        o = self.i_layer(x)
        o = self.h_layers(o)
        return self.o_layer(o)

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().numpy()


class DataIterator(object):
    def __init__(self, dataset, batchsize, device):
        # object reference (a list of tuples of lists)
        self._dataset = dataset
        self._len = len(dataset[0])
        # keep track of current index
        self._index = 0
        # the batch size
        self._batchsize = batchsize
        self._device = device

    def __next__(self):
        ''''Returns the next value from object's lists '''

        n = min(self._len, self._index + self._batchsize)
        if self._index < self._len:
            x = self._dataset[0][self._index:n]
            y = self._dataset[1][self._index:n]
            self._index = n
            return Ten(x).to(self._device), Ten(y).to(self._device)

        # End of Iteration
        raise StopIteration


class Dataset(object):

    def __init__(self, params, mode, device):
        assert mode in ['train', 'test', 'valid']
        np.random.seed(params['seed'])
        #self._const = 0  # constrain counter
        self._device = device
        self._n_data = params['n_data']
        self._benchmark = params['benchmark']
        self._batchsize = params['batch_size']
        self._violated_const_ratio = params['violated_const_ratio'] if mode == 'train' else 0
        # builds ad hoc dataset, the number of violated_ constraints can be tuned
        (X, y) =  build_dataset(self._benchmark, self._n_data, self._violated_const_ratio, params['seed'])
        #self._const += len(violated_const(X, y))
        self._n_var = len(X[0])
        # select data
        indices = self._get_indexes(params, self._n_data, mode, params['seed'])
        X, y = X[indices], y[indices]
        self._dataset = tuple([X, y])

    @property
    def n_var(self):
        return self._n_var

    #@property
    #def const(self):
    #    return self._const

    def _get_indexes(self, params, n_data, mode, seed):
        indices = np.arange(n_data)
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_size = dict()
        modeidx = {'train': 0, 'test': 1, 'valid': 2}
        for m in ['train', 'test', 'valid']:
            split_size[m] = int(params['split'][modeidx[m]] * n_data)
        if mode == 'train':
            indices = indices[0:split_size['train']]
        elif mode == 'test':
            indices = indices[split_size['train']:split_size['test'] + split_size['train']]
        else:
            indices = indices[split_size['train'] + split_size['test']:-1]
        return indices

    def __iter__(self):
        return DataIterator(self._dataset, self._batchsize, self._device)



class AbstractAgent():
    def __init__(self, params, d_train, d_test, d_val, start_point_seed=0):
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if use_cuda else "cpu")

        self._train_data = d_train
        self._test_data = d_test
        self._valid_data = d_val
        self.start_point_seed = start_point_seed
        #torch.manual_seed(start_point_seed)
        self._nepochs = params['epochs']
        self._batchsize = params['batch_size']

        #self._model = None
        self._optimizer = None
        self._loss = None
        self.logs = []
        self.verbose = False

    def _init_model(self):

        torch.manual_seed(self.start_point_seed)
        net_par = {'i_dim': self._train_data.n_var,
                   'o_dim': 1,
                   'h_dim': 10,
                   'n_layers': 1}

        self._model = Net(net_par)
        self._optimizers = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self._LR_multiplier_list = []


    def train(self):
        for epoch in range(self._nepochs):
            for (x, y) in self._train_data:
                y_pred = self.predict(x)
                loss = self.compute_loss(y_pred, y, x)
                self.propagate_loss(loss)
            self.print_report(epoch)
            self.validation_step(epoch)

    def opt_lr_rate(self):
        '''
        Optimize the Lagrangian step size. Should we run only 1 time for data of same violation ratio, 
        and  same number of training samples,

        :return: set class object with optimal logs, and optimal model with optimal lr 
        '''
        model_list = []
        lr_list = [1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]
        val_mae_list = []
        val_vc_list = []
        model_list = []
        logs_list = []
        _LR_multiplier_list_list = []
        for lr in lr_list:
            self._LR_rate = lr
            self._init_model()
            self.train(options = {'mult_fixed':False})
            val_mae_list.append(copy.deepcopy(self.logs[-1][0]))  # use deep copy for safety reason
            val_vc_list.append(copy.deepcopy(self.logs[-1][2]))
            model_list.append(copy.deepcopy(self._model))
            logs_list.append(copy.deepcopy(self.logs))
            _LR_multiplier_list_list.append(copy.deepcopy(self._LR_multiplier_list))

        self.model_list = copy.deepcopy(model_list)
        self.val_mae_list = copy.deepcopy(val_mae_list)
        self.val_vc_list = copy.deepcopy(val_vc_list)

        val_mae_list = [(x - min(val_mae_list)) / float(max(val_mae_list)) for x in val_mae_list]
        if max(val_vc_list)>0:
            val_vc_list = [(x - min(val_vc_list)) / float(max(val_vc_list)) for x in val_vc_list]
        metric_list = val_mae_list + val_vc_list

        best_index = [idx for idx in range(5) if metric_list[idx] == min(metric_list)][0]

        self._model = model_list[best_index]
        self.logs = logs_list[best_index]
        self._LR_rate = lr_list[best_index]
        self.logs_list = logs_list
        self._LR_multiplier_list_list = _LR_multiplier_list_list
        self._LR_multiplier_list = _LR_multiplier_list_list[best_index]


    def predict(self, x):
        return self._model(x)

    def propagate_loss(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def validation_step(self, epoch):
        #mae_res = list()
        violated_const_model = 0
        violated_const_dataset = 0
        (X, y) = self._valid_data._dataset
        y_pred = self._model.predict(Ten(X))
        val_mae = mae(y, y_pred)
        violated_const_dataset += len(violated_const(X, y))
        violated_pairs = violated_const(X, y_pred)
        violated_const_model += len(violated_pairs)

        y_pred = copy.deepcopy( np.array([10 ** -t for t in y_pred]))
        y_true = copy.deepcopy( np.array([10 ** -t for t in y]))

        val_rmse = np.sqrt( mse(y_true, y_pred))
        y_diff = np.abs(y_true - y_pred)
        sum_mag_viol = np.sum([abs( y_pred[i] -y_pred[j]) for (i,j) in violated_pairs]  )
        median_ae = np.median(y_diff)
        mean_ae = np.mean(y_diff)

        if self.verbose:
            print(f"epoch: {epoch}, "
                  f"MAE: {val_mae}, "
                  f"Violated constraints dataset: {violated_const_dataset}, "
                  f"Violated constraints model: {violated_const_model}")

        self.logs.append([val_mae, violated_const_dataset,  violated_const_model, median_ae, mean_ae, val_rmse, sum_mag_viol])
    def print_report(self, epoch):
        pass

    def test(self):
        #mae_list = list()
        violated_const_dataset = 0
        violated_const_model = 0
        (X, y) = self._test_data._dataset
        y_pred = self._model.predict(Ten(X))

        test_mae = mae(y, y_pred)
        violated_const_dataset += len(violated_const(X, y))
        violated_pairs = violated_const(X, y_pred)
        violated_const_model += len(violated_pairs)

        y_pred = copy.deepcopy(np.array([10 ** -t for t in y_pred]))
        y_true = copy.deepcopy(np.array([10 ** -t for t in y]))

        y_diff = np.abs(y_true - y_pred)
        median_ae = np.median(y_diff)
        mean_ae = np.mean(y_diff)
        rmse_ = np.sqrt(mse(y, y_pred))
        sum_mag_viol = np.sum([abs(y_pred[i] - y_pred[j]) for (i, j) in violated_pairs])

        #print(f"Test precision: {test_mae}, "
        #      f"Violated constraints Dataset: {violated_const_dataset}, "
       #       f"Violated constraints Model: {violated_const_model}, "
        #      f"Duplicates: {duplicates(y_pred)}")
        self.y_true  = y_true
        self.y_pred = y_pred
        self.violated_pairs = violated_pairs
        return (test_mae, violated_const_model, violated_const_dataset, median_ae, mean_ae, rmse_, sum_mag_viol )
        #return (test_mae, violated_const_model, median_ae, mean_ae, rmse_)


    def build_kwb_matrix(self, data):
        """ Build matrix containg dominance informatins

        Every couple (dominant, dominated) is tracked in the matrix.
        Each row represents a couple, while each column a training sample.
        Dominand samples are marked with -1, while the dominated with 1.
        """
        all_couples = couples(data.tolist())
        n = max(1, len(all_couples))
        kwb_matrix = np.zeros((n, len(data)))
        for (k, (i, j)) in enumerate(all_couples):
            kwb_matrix[k, i] = -1
            kwb_matrix[k, j] = 1
        return Ten(kwb_matrix.T)

    def compute_loss(self, y_pred, y, x=None):
        pass

    def plot(self):
        pass


class Regressor(AbstractAgent):
    def __init__(self, params, d_train, d_test, d_val, start_point_seed):
        super(Regressor, self).__init__(params,d_train, d_test, d_val,  start_point_seed)
        #self._train_data = d_train
        #net_par = {'i_dim': self._train_data.n_var,
        #           'o_dim': 1,
        #           'h_dim': 10,
         #          'n_layers': 1}
        #torch.manual_seed(start_point_seed)
        #self._model = Net(net_par)
        # Optimizers, for each dataset partition
        #self._optimizers = torch.optim.Adam(self._model.parameters(), lr=0.001)
        # Loss functions for each dataset partition
        self._loss = nn.MSELoss()

    def train(self):
        super()._init_model()

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


class SBRregressor(AbstractAgent):
    def __init__(self, params, d_train, d_test, d_val, start_point_seed):
        super(SBRregressor, self).__init__(params, d_train, d_test, d_val, start_point_seed)

        #self._train_data = d_train
        #net_par = {'i_dim': self._train_data.n_var,
        #           'o_dim': 1,
        #           'h_dim': 10,
         #          'n_layers': 1}

        #torch.manual_seed(start_point_seed)
        #self._model = Net(net_par)

        # Optimizers, for each dataset partition
        #self._optimizers = torch.optim.Adam(self._model.parameters(), lr=0.001)
        # Loss functions for each dataset partition
        self._loss = nn.MSELoss()
        self._const_avg_batch = 0
        self.count_const()
        self._LR_rate = None

        self._violations_epoch = []
        self._LR_multiplier_list = []



    def count_const(self):
        const_batch = []
        for (x, y) in self._train_data:
            n = max(1, len(couples(x.tolist())))
            const_batch.append(n)
        self.const_avg_batch = int(np.mean(np.array(const_batch)))

    # @property
    def const_avg_batch(self):
        return self.const_avg_batch

    def train(self, options):

        super()._init_model()

        if options['mult_fixed']:
            self._LR_multiplier = 1
        else:
            self._LR_multiplier = 0

        for epoch in range(self._nepochs):
            violations_epoch = []
            for (x, y) in self._train_data:
                M = self.build_kwb_matrix(x)
                y_pred = self.predict(x)
                loss, violation = self.compute_loss(M, y_pred, y, x)
                self.propagate_loss(loss)
                violations_epoch.append(copy.deepcopy(violation))
            if not options['mult_fixed']:
                self.update_LR_multipliers(violations_epoch)
            self.print_report(epoch)
            self.validation_step(epoch)


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
        self._LR_multiplier_list.append(copy.deepcopy(self._LR_multipliers))
        pass
        #print('\t LR mult:', self._LR_multiplier)


class SBRregressor2(AbstractAgent):
    def __init__(self, params, d_train, d_test, d_val, start_point_seed):
        super(SBRregressor2, self).__init__(params, d_train, d_test, d_val, start_point_seed)

        #self._train_data = d_train
        #net_par = {'i_dim': self._train_data.n_var,
        #          'o_dim': 1,
        #            'h_dim': 10,
        #           'n_layers': 1}

        #torch.manual_seed(start_point_seed)
        #self._model = Net(net_par)
        #self._optimizers = torch.optim.Adam(self._model.parameters(), lr=0.001)
        # Loss functions for each dataset partition
        self._loss = nn.MSELoss()
        #self.initialize_LR_multipliers()
        self._LR_rate =  None
        self._violations_epoch = []
        self.const_avg_batch = 0
        self._LR_multiplier_list = []


    def initialize_LR_multipliers(self, options):
        self._LR_multipliers = []
        const_batch = []
        for (x, y) in self._train_data:
            n = max(1, len(couples(x.tolist())))
            const_batch.append(n)
            if options['mult_fixed']:
                # all Lagrangian multipliers set to be one, and will not be updated during optimization
                self._LR_multipliers.append(np.ones(n))
            else:
                self._LR_multipliers.append(np.zeros(n))
        self.const_avg_batch = int(np.mean(np.array(const_batch)))
        self._LR_multipliers = np.concatenate(self._LR_multipliers)

    # @property
    def const_avg_batch(self):
        return self.const_avg_batch()

    def train(self, options):

        super()._init_model()

        self.initialize_LR_multipliers(options)

        for epoch in range(self._nepochs):
            violations_epoch = []
            offset = 0
            for (x, y) in self._train_data:
                M = self.build_kwb_matrix(x)
                y_pred = self.predict(x)
                loss, violations = self.compute_loss(M, offset, y_pred, y, x)
                self.propagate_loss(loss)
                violations_epoch += violations
                offset += len(violations)
            if not options['mult_fixed']:
                self.update_LR_multipliers(violations_epoch)

            self.validation_step(epoch)
            self.print_report(epoch)

    ''' Override the compute_loss method in AbstractAgent class'''

    def compute_loss(self, M, offset, y_pred, y, x=None):
        loss = self._loss(y_pred, y)
        # g(x') - g(x'') where g(x'') is dominant over g(x'), hence should be have grater value,
        # (N.B. we are considering the -log10 of the error otherwise it would be g(x'') - g(x')
        rules = torch.mm(torch.transpose(M, 0, 1), y_pred)
        # filter only the positive values (violations), therefore the violations to the constraint
        rules = torch.max(Ten(np.zeros(rules.size()[0])), torch.transpose(rules, 0, 1))[0]
        # LR * max(0, g(x') - g(x''))
        lr_mults = Ten([self._LR_multipliers[offset + i] for i in range(len(rules))])
        loss += torch.sum(lr_mults * rules)
        # The below code seems  a bug, ignoring the contribution of violated constraints during optimization
        #
        #loss += torch.sum(Ten([self._LR_multipliers[offset + i] * rules[i] for i in range(len(rules))]))
        violations = [x.item() for x in rules]

        return loss, violations

    ''' Override the propagate_loss method in AbstractAgent class'''

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
        for i in range(len(violations)):
            self._LR_multipliers[i] = self._LR_multipliers[i] + (self._LR_rate * violations[i])

    def print_report(self, epoch):
        self._LR_multiplier_list.append(copy.deepcopy(self._LR_multipliers))
        # a single element contains a vector of lambda_{i,j} at epoch time t
        pass

        #print('\t LR mult:', self._LR_multipliers)
        #print('\t AVG LR mult:', np.sum(self._LR_multipliers))



def test(benchmark, violated_const_ratio, test_seed):



    res = {'test_seed':test_seed} # store all model results
    params = {'epochs': 150,
                   'n_data': 4000,
                   'batch_size': 256,
                   'violated_const_ratio': violated_const_ratio,  # this is used to create a trainig set with a specific
                   # amount of contraint violations
                   'benchmark': benchmark,
                   'split': [0.5, 0.25, 0.25],
                   'seed': test_seed}

    d_trainall = Dataset(params, 'train', 'cpu')
    d_test = Dataset(params, 'test', 'cpu')
    d_valall = Dataset(params, 'valid', 'cpu')
    res['d_test'] = d_test
    X_test, y_test = d_test._dataset

    val_size = 300 # fix validation set size
    for train_size in [200, 400, 600, 800, 1000]:
        res[train_size] = {}
        for split_seed in range(20):
            print('train size = {} , seed = {}'.format(train_size, split_seed))
            np.random.seed(split_seed)
            X_train,y_train  = copy.deepcopy( d_trainall._dataset)
            idx_train = np.random.choice( list(range(len(X_train))), train_size)
            X_train, y_train = X_train[idx_train,:], y_train[idx_train]

            d_train = copy.deepcopy(d_trainall)
            d_train._dataset = (X_train, y_train)
            y_med_pred = np.median(y_train)* np.ones(len(y_test))

            X_val, y_val = copy.deepcopy(d_valall._dataset)
            idx_val = np.random.choice(list(range(len(X_val))), val_size)
            X_val, y_val = X_val[idx_val,:], y_val[idx_val]
            d_val = copy.deepcopy(d_valall)
            d_val._dataset = (X_val, y_val)
            res[train_size][split_seed] ={'d_val':d_val, 'd_train': d_train}
            #model_1_mae, model_2_mae, model_2_1_mae, model_3_mae, model_3_1_mae = [], [], [], [], []

            #model_1_violated_const, model_2_violated_const, model_3_violated_const = [], [], []
            #model_2_1_violated_const, model_3_1_violated_const = [], []
            #sum_mag_viol_list_1, sum_mag_viol_list_2_1, sum_mag_viol_list_2, sum_mag_viol_list_3_1, sum_mag_viol_list_3 = [], [], [], [], []

            model_1 = Regressor(params, d_train, d_test, d_val, 0) # 0 is random seed of pytorch make sure all models starting from the same point
            model_1.train()
            tmp = model_1.test()
            res[train_size][split_seed]['model_1_perf'] = copy.deepcopy(tmp)
            #model_1_mae.append(tmp[0])
            #model_1_violated_const.append(tmp[1])
            #sum_mag_viol_list_1.append(tmp[-1])

            # regularization with single multiplier =1
            model_2_1 = SBRregressor(params, d_train, d_test, d_val, 0)
            model_2_1.train(options = {'mult_fixed':True})
            tmp = model_2_1.test()
            res[train_size][split_seed]['model_2_1_perf'] = copy.deepcopy(tmp)
            # model_2_1_mae.append(tmp[0])
            # model_2_1_violated_const.append(tmp[1])
            # sum_mag_viol_list_2_1.append(tmp[-1])

            # regularization with a multiplier for each constraint, each multiplier has value  = 1
            model_3_1 = SBRregressor2(params, d_train, d_test, d_val, 0)
            model_3_1.train(options = {'mult_fixed':True})
            tmp = model_3_1.test()
            res[train_size][split_seed]['model_3_1_perf'] = copy.deepcopy(tmp)
            # model_3_1_mae.append(tmp[0])
            # model_3_1_violated_const.append(tmp[1])
            # sum_mag_viol_list_3_1.append(tmp[-1])


            ###################regularization with single multiplier updated gradually, starts with 0

            model_2 = SBRregressor(params, d_train, d_test, d_val, 0)
            if split_seed == 0:
                model_2.opt_lr_rate()
                best_lr_model_2 = copy.deepcopy( model_2._LR_rate)
            else:
                model_2._LR_rate = copy.deepcopy( best_lr_model_2)
                model_2.train(options={'mult_fixed': False})
            tmp = model_2.test()
            res[train_size][split_seed]['model_2_perf'] = copy.deepcopy(tmp)
            # model_2_mae.append(tmp[0])
            # model_2_violated_const.append(tmp[1])
            # sum_mag_viol_list_2.append(tmp[-1])


            ########## regularization with a multiplier for each constraint, each multiplier has updated gradually, starts with 0


            model_3 = SBRregressor2(params, d_train, d_test, d_val, 0)
            if split_seed == 0:
                model_3.opt_lr_rate()
                best_lr_model_3 = copy.deepcopy(model_3._LR_rate)
            else:
                model_3._LR_rate = copy.deepcopy( best_lr_model_3)
                model_3.train(options={'mult_fixed': False})

            tmp = model_3.test()
            res[train_size][split_seed]['model_3_perf'] = copy.deepcopy(tmp)
            # model_3_mae.append(tmp[0])
            # model_3_violated_const.append(tmp[1])
            # sum_mag_viol_list_3.append(tmp[-1])

            res[train_size][split_seed]['model_3'] = copy.deepcopy(model_3)
            res[train_size][split_seed]['model_2'] = copy.deepcopy( model_2)
            res[train_size][split_seed]['model_1'] = copy.deepcopy( model_1)
            res[train_size][split_seed]['model_3_1'] = copy.deepcopy( model_3_1)
            res[train_size][split_seed]['model_2_1'] = copy.deepcopy(model_2_1)
            res[train_size][split_seed]['dump_model_perf'] = copy.deepcopy(mae(y_test, y_med_pred))

    filename = str(benchmark) + '_test_seed_{}'.format(test_seed)+\
                    "_vconst" + str(violated_const_ratio) +'.pkl'

    file_handle = open(path_to_repo + 'results/' +filename, 'wb')
    pickle.dump(res, file_handle)
