"""Shared pieces of cnn_cmip.py and cnn_trans.py.

Both scripts train the same CNN on the 17 CMIP models (inp_cmip17.npy, pr_cmip17.npy).
Only the ensemble size and the training procedure differ, so the data preparation,
the model definition and the target loops live here.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Input data directory, the "output" folder next to the current working directory
INPUT_DIR = os.path.join(os.path.dirname(os.getcwd()), 'output')

# Target months (m = 5 ... 10) and their names
M_LIST = np.arange(5, 11)
SEA_LIST = ['may', 'jun', 'jul', 'aug', 'sep', 'oct']
# gap = 4, 5, 6 corresponds to leadtime 1, 2, 3 month (gap - 3)
GAP_LIST = np.arange(4, 7)


def load(name):
    return np.load(os.path.join(INPUT_DIR, name))


def norm(x):
    return (x - x.mean()) / x.std()


def is_may_lead3(m, gap):
    # m - gap is negative only for May with gap 6, which needs the *53 variants
    return m == 5 and gap == 6


def targets():
    """Yield (m, sea, gap) in the same order as the original nested loops."""
    for m, sea in zip(M_LIST, SEA_LIST):
        for gap in GAP_LIST:
            yield m, sea, gap


class CnnBase:
    ensemble = 5

    def __init__(self, m, sea, gap):
        self.m = m
        self.sea = sea
        self.gap = gap
        self.Mon = self.m - self.gap
        self.height, self.width, self.channels = 24, 72, 6
        self.gcm = 1725
        self.gcm_n = 17
        self.epochs = 20
        self.batch_size = 200
        self.learnrate = 0.0001
        self.loss_function = 'mse'
        self.metrics = 'mae'
        self.activate_function = 'tanh'
        self.C, self.H = 30, 30
        self.vsample = 400

    def _cmip_slices(self, x, y, gcm_mon, x_offset, y_offset):
        total = self.gcm_n * gcm_mon
        inp = np.empty((total, self.height, self.width, self.channels))
        out = np.empty(total)
        for i in range(self.gcm_n):
            inp[i*gcm_mon:(i+1)*gcm_mon, :, :, :] = x[i*self.gcm+x_offset:(i+1)*self.gcm:12]
            out[i*gcm_mon:(i+1)*gcm_mon] = y[i*self.gcm+y_offset:(i+1)*self.gcm:12]
        out = norm(out)
        return inp, out

    def cmipMon(self, x, y):
        return self._cmip_slices(x, y, int(self.gcm/12)+1, self.Mon, self.m-4)

    def cmipMon53(self, x, y):
        return self._cmip_slices(x, y, int(self.gcm/12), 11, 13)

    def shuffle(self, inputData, outputData):
        random_number = len(inputData)
        where_trainShuffle = random_number - self.vsample
        train_index = np.random.choice(random_number, where_trainShuffle, replace=False)
        x_tr = inputData[train_index]
        y_tr = outputData[train_index]

        val_index = np.random.choice(self.vsample, self.vsample, replace=False)
        x_v = np.delete(inputData, train_index, 0)
        y_v = np.delete(outputData, train_index)
        x_v = x_v[val_index]
        y_v = y_v[val_index]
        return x_tr, y_tr, x_v, y_v

    def checkMask(self, data):
        plt.imshow(data[0, :, :, 0], cmap='RdBu_r')
        plt.clim(-1.5, 1.5)
        plt.colorbar(orientation='horizontal')
        plt.show()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(self.C, (4, 8), activation=self.activate_function, input_shape=(self.height, self.width, self.channels), padding='SAME'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(self.C, (2, 4), activation=self.activate_function, padding='SAME'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(self.C, (2, 4), activation=self.activate_function, padding='SAME'))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.H, activation=self.activate_function))
        model.add(layers.Dense(1, activation='linear'))

        optimizer = tf.keras.optimizers.RMSprop(self.learnrate)

        model.compile(optimizer=optimizer, loss=self.loss_function, metrics=[self.metrics])
        return model


def print_summary(corr_ensemble, rmse_ensemble):
    print('------------------------------------------------------')
    print('Score per ensemble')
    for i in range(len(corr_ensemble)):
        print('-----------------------------------------------------')
        print(f'> Fold {i+1} - RMSE: {rmse_ensemble[i]} - Corr: {corr_ensemble[i]}')
    print('-----------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Corr: {np.mean(corr_ensemble)} (+- {np.std(corr_ensemble)}')
    print(f'> RMSE: {np.mean(rmse_ensemble)} (+- {np.std(rmse_ensemble)}')
    print('-----------------------------------------------------')
