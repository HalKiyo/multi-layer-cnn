import os
import bisect
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, losses

def main():
    # import data
    path = '/docker/mnt/d/research/MtoD/output'
    load = lambda x:np.load(os.path.join(path,x))
    inp_cmip = load('inp_cmip17.npy')
    pr_cmip = load('pr_cmip17.npy')

    # forecast target setting
    cnn = cnn_cmip(8, 'aug', 4) # 1 month forecast for August using May, Jun, July env

    # Load data
    inp, out = cnn.cmipMon(inp_cmip, pr_cmip) # inp(2448, 24, 72, 6), out(2448,)
    out_bnds, out_class = cnn.EFD(out) # out_bnds(48, 2), out_class(2448,)

    # plot histgram
    cnn.draw_hist(out, out_bnds)
    exit()

    # Preprocess train and validation data
    inp_masked = ma.masked_where(inp>9999, inp)
    x_train, y_train, x_val, y_val = cnn.shuffle(inp_masked, out_class)
    y_train_bn, y_val_bn = cnn.class_to_onehot(y_train, y_val)

    # Train and validate model
    model = cnn.build_model()
    cnn.train(model, x_train, y_train_bn, x_val, y_val_bn)

class cnn_cmip:
    def __init__(self, m, sea, gap):
        self.m = m # month range from 5 to 10(May to October)
        self.sea = sea # ['may','jun','jul','aug','sep','oct'] strig list for filename
        self.gap = gap # leadtime-gap ranging from 4 to 6(1month, 2month, 3month ahead
        self.Mon = m - gap # index for target month in data source(x)
                           # if target=6(Jun), gap4=Mon(Mar, gap5=Mon(Feb, gap6=Mon(Jan
                           # 1month: Mar,Apr,May, 2month: Feb,Mar,Apr, 3month: Jan,Feb,Mar

        self.gcm = 1725 # number of data in a gcm
        self.gcm_n = 17 # number of gcms
        self.gcm_mon = int(self.gcm/12) + 1 # number of years in a gcm
        self.total = self.gcm_n * self.gcm_mon # number of samples for a month target

        # 48nb_class * 51samples = 2448
        mult = 12
        self.batch_sample = 51*mult
        self.nb_class = int(48/mult)

        self.height, self.width, self.channels = 24, 72, 6 # shape of input images
        self.C, self.H = 64, 64 # hidden layer sizes
        self.vsample = 200 # number of validation samples
        self.epochs = 25 # number of epochs
        self.batch_size = 200 # number of batch_size
        self.learnrate = 0.00001 # learnrate of optimizers("RMSprop")
        self.optimizer = tf.keras.optimizers.Adam(self.learnrate)
        self.loss_function = 'categorical_crossentropy' # loss function
        self.metrics = 'accuracy' # mean absolute error for validation metric
        self.activate_function = 'relu' # activation function

    def norm(self, x):
        return ( (x - x.mean() ) / x.std() )

    def cmipMon(self, x, y):
        """ Extracting training dataset of x, y(target month) exept for May3month """

        inp = np.empty( (self.total, self.height, self.width, self.channels) )
        out = np.empty(self.total)
        for i in range(self.gcm_n):
            inp[i * self.gcm_mon : (i + 1) * self.gcm_mon, :, :, :] = \
                x[i * self.gcm + self.Mon : (i+1) * self.gcm : 12]
            out[i * self.gcm_mon : (i + 1) * self.gcm_mon] = \
                y[i * self.gcm + (self.m - 4) : (i + 1) * self.gcm : 12]
        out = self.norm(out)

        return inp, out

    def EFD(self, out):
        out_sorted = np.sort(out) # shape = (2448,)
        out_bnd = [ out_sorted[i] for i in range(0, len(out_sorted), self.batch_sample) ]

        out_class = np.empty( len(out_sorted) ) # shape = (2448,)

        for i, value in enumerate(out):
            # out_bnd = [ min, min+1, ..., max-1, max]
            # index = 0,| 1, 2, ..., len(out_bnd) - 2, len(out_bnd) - 1, len(out_bnd)|
            # default: if x == a, x comes rightside of a
            # because class_to_onehot requires index less than 48, (label - 1) is applied
            label = bisect.bisect(out_bnd, value)
            out_class[i] = int(label - 1)

        out_bnd.append(out_sorted[-1])
        out_bnd = np.array(out_bnd) # shape = (49,)
        out_bnds = np.empty( ( len(out_bnd) - 1, 2 ) ) # shape = (48,)

        for i in range(len(out_bnds)):
            # out_bnds corresponds label string for rain intensity range
            out_bnds[i, 0] = out_bnd[i]
            out_bnds[i, 1] = out_bnd[i + 1]

        return out_bnds, out_class

    def class_to_onehot(self, y_train, y_val):
        y_train_bn = [ tf.keras.utils.to_categorical(i, self.nb_class) for i in y_train ]
        y_train_bn = np.array(y_train_bn) # (2048, 48)
        y_val_bn = [ tf.keras.utils.to_categorical(i, self.nb_class) for i in y_val ]
        y_val_bn = np.array(y_val_bn) # (200, 48)

        return y_train_bn, y_val_bn

    def shuffle(self, inputData, outputData):
        """ Shuffle the data following validation sample number """

        random_number = len(inputData)
        where_trainShuffle = random_number - self.vsample
        train_index = np.random.choice(random_number,where_trainShuffle,replace=False)
        x_tr = inputData[train_index]
        y_tr = outputData[train_index]

        val_index = np.random.choice(self.vsample,self.vsample,replace = False)
        x_v = np.delete(inputData,train_index,0)
        y_v = np.delete(outputData,train_index)
        x_v = x_v[val_index]
        y_v = y_v[val_index]

        return x_tr,y_tr,x_v,y_v

    def build_model(self):
        """ Building the model structure """

        model = models.Sequential()
        model.add(layers.Conv2D(self.C, (4,8), activation=self.activate_function,
            input_shape=(self.height,self.width,self.channels), padding='SAME'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(self.C, (2,4), activation=self.activate_function, padding='SAME'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(self.C, (2,4), activation=self.activate_function, padding='SAME'))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.H, activation=self.activate_function))
        model.add(layers.Dense(self.nb_class, activation='softmax'))

        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=[self.metrics])

        return model

    def train(self, model, x_train, y_train, x_val, y_val):
        """ Training the model with training dataset """

        model.fit(x_train, y_train,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  verbose=2)

        pred = model.predict(x_val)
        print(pred)
        exit()

        model.evaluate(x_val,  y_val, verbose=2)

        return pred

    def draw_hist(self, data, class_bnds):
        plt.style.use('fivethirtyeight')

        fig = plt.figure()

        ax = plt.subplot()
        ax.hist(data, bins=100, alpha=.5, color='darkcyan')

        for i in class_bnds:
            ax.axvline(i[0], ymin=0, ymax=self.batch_size*self.nb_class, alpha=.8, color='salmon')

        ax2 = ax.twinx()
        sns.kdeplot(data=data, ax=ax2, color='sandybrown')

        plt.show()

    def draw_val(self, val, data, class_bnds):
        plt.style.use('fivethirtyeight')

        fig = plt.figure()

        ax = plt.subplot()
        ax.hist(data, bins=100, alpha=.5, color='darkcyan')

        for i in class_bnds:
            ax.axvline(i[0], ymin=0, ymax=self.batch_size*self.nb_class, alpha=.8, color='salmon')

        ax2 = ax.twinx()
        sns.kdeplot(data=data, ax=ax2, color='sandybrown')

        plt.show()


if __name__ == '__main__':
    main()
