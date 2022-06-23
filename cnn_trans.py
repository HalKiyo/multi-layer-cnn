from __future__ import absolute_import,division, print_function, unicode_literals
import os 
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, optimizers, losses


class cnn_trans:
    def __init__(self,m,sea,gap):
        self.m = m
        self.sea = sea
        self.gap = gap
        self.Mon = self.m - self.gap
        self.height,self.width,self.channels = 24,72,6
        self.gcm = 1725
        self.gcm_n = 17
        self.ensemble = 20
        self.epochs = 20
        self.batch_size = 200
        self.learnrate = 0.0001
        self.loss_function = 'mse'
        self.metrics = 'mae'
        self.activate_function = 'tanh'
        self.C,self.H = 30,30
        self.vsample = 400

    def cmipMon(self,x,y):
        gcm_mon = int(self.gcm/12)+1
        total = self.gcm_n*gcm_mon
        inp = np.empty((total,self.height,self.width,self.channels))
        out = np.empty(total)
        for i in range(self.gcm_n):
            inp[i*gcm_mon:(i+1)*gcm_mon,:,:,:] = x[i*self.gcm+self.Mon:(i+1)*self.gcm:12]
            out[i*gcm_mon:(i+1)*gcm_mon] = y[i*self.gcm+(self.m-4):(i+1)*self.gcm:12]
        out = norm(out)
        return inp,out

    def cmipMon53(self,x,y):
        gcm_mon = int(self.gcm/12)
        total = self.gcm_n*gcm_mon
        inp = np.empty((total,self.height,self.width,self.channels))
        out = np.empty(total)
        for i in range(self.gcm_n):
            inp[i*gcm_mon:(i+1)*gcm_mon,:,:,:] = x[i*self.gcm+11:(i+1)*self.gcm:12]
            out[i*gcm_mon:(i+1)*gcm_mon] = y[i*self.gcm+13:(i+1)*self.gcm:12]
        out = norm(out)
        return inp,out

    def transDset(self,aphro,sode,godas):
        ap_tr,ap_te = aphro[(self.m-1):42*12],aphro[42*12+(self.m-1):]
        x_trans,y_trans = soda[self.Mon::12,:,:,:],norm(ap_tr[::12])
        x_test,y_test = godas[self.Mon::12],norm(ap_te[::12])
        return x_trans,y_trans,x_test,y_test

    def transDset53(self,aphro,soda,godas):
        ap_tr,ap_te = aphro[(11+5):42*12],aphro[42*12+11+5:]
        x_trans,y_trans = soda[11::12,:,:,:],norm(ap_tr[::12])
        x_test,y_test = godas[11::12],norm(ap_te[::12])
        return x_trans, y_trans,x_test,y_test

    def checkMask(self,data):
        plt.imshow(data[0,:,:,0],cmap = 'RdBu_r')
        plt.clim(-1.5,1.5)
        plt.colorbar(orientation='horizontal')
        plt.show()

    def checkPr(self,data):
        plt.plot(mp.arange(len(data)),data)
        plt.ylim()
        pt.show()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(self.C,(4,8),activation=self.activate_function, input_shape=(self.height,self.width,self.channels),padding='SAME'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(self.C,(2,4),activation=self.activate_function,padding='SAME'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(self.C,(2,4),activation=self.activate_function,padding='SAME'))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.H, activation=self.activate_function))
        model.add(layers.Dense(1, activation='linear'))

        optimizer = tf.keras.optimizers.RMSprop(self.learnrate)

        model.compile(optimizer=optimizer,
                loss = self.loss_function,
                metrics = [self.metrics]
                )
        return model

    def shuffle(self,inputData,outputData,vsample=400):
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

    def main(self,inp,out,x_trans,y_trans,x_test,y_test):
        cnn_ensemble = []
        corr_ensemble = []
        rmse_ensemble = []
        for i in range(self.ensemble):
            print('----------------------------------------------------')
            print(f'Training for ensemble {i} ...')
            base_model = self.build_model()
            checkpoint_path = "training_"+self.sea+"/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                    save_weights_only=True,
                    varbose=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
            inp2 = ma.masked_where(inp>9999,inp)
            x_train,y_train,x_val,y_val = self.shuffle(inp2,out)

            history = base_model.fit(x_train,
                    y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(x_val,y_val),
                    verbose =1,
                    callbacks = [early_stop,cp_callback]
                    )

            latest = tf.train.latest_checkpoint(checkpoint_dir)

            trans_model = self.build_model()
            trans_model.load_weights(latest)
            trans_model.layers[0].trainable = False
            trans_model.layers[1].trainable = False
            trans_model.layers[2].trainable = False
            trans_model.layers[3].trainable = False
            trans_model.layers[4].trainable = False

            optimizer = tf.keras.optimizers.RMSprop(0.0001)
            trans_model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])
            early_stop2 = keras.callbacks.EarlyStopping(monitor='val_loss',patience=500)
            history2 = trans_model.fit(x_trans,y_trans,
                    epochs=50,
                    validation_data=(x_test,y_test),
                    verbose=1,
                    callbacks = [early_stop2]
                    )

            pred = trans_model.predict(x_test)
            corr = np.corrcoef(pred[:,0],y_test)[0,1]
            scores = trans_model.evaluate(x_test,y_test,verbose=0)
            rmse = np.sqrt(scores[0])
            print(f'Score for ensemble {i}: RMSE of {rmse}; corr of {corr}')
            cnn_ensemble.append(pred)
            corr_ensemble.append(corr)
            rmse_ensemble.append(np.sqrt(scores[0]))

        print('------------------------------------------------------------')
        print('Score per ensemble')
        for i in range(0, len(corr_ensemble)):
            print("----------------------------------------------------")
            print(f'> Fold{i+1} - RMSE: {rmse_ensemble[i]} - Corr: {corr_ensemble[i]}')
        print('--------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Corr: {np.mean(corr_ensemble)} (+- {np.std(corr_ensemble)}')
        print(f'> RMSE: {np.mean(rmse_ensemble)} (+- {np.std(rmse_ensemble)}')
        print("--------------------------------------------")

        spath = '/home/Hasegawa/Dthesis/water/output/cnn_trans/leadtime_'+np.str(self.gap-3)+'month'
        os.makedirs(spath,exist_ok=True)
        sfile = os.path.join(spath,'trans'+self.sea+'.npy')
#		np.save(sfile,cnn_ensemble)

if __name__ == '__main__':
    print('Now Loading ...')
    path = os.path.join(os.path.dirname(os.getcwd()),'output')
    load = lambda x:np.load(os.path.join(path,x))
    norm = lambda x:((x-x.mean())/x.std())

    inp_cmip = load('inp_cmip17.npy')
    pr_cmip  = load('pr_cmip17.npy')
    aphro    = load('aphro_month_1951-2015.npy')
    godas    = load('inp_godas.npy')
    soda     = load('soda_train_for_pr.npy')
    soda = ma.masked_where(soda>9999,soda)
    godas = ma.masked_where(godas>9999,godas)

    m_list = np.arange(5,11)
    sea_list = ['may','jun','jul','aug','sep','oct']
    gap_list = np.arange(4,7)

    for (a,b) in zip (m_list,sea_list):
        for c in gap_list:
            if a == 5 and c == 6:
                cnn = cnn_trans(a,b,c)
                print(a,b,c)
                inp,out = cnn.cmipMon53(inp_cmip,pr_cmip)
                x_trans,y_trans,x_test,y_test = cnn.transDset53(aphro,soda,godas)
                print(len(x_trans),len(y_trans),len(x_test),len(y_test))
                cnn.main(inp,out,x_trans,y_trans,x_test,y_test)
            else:
                cnn = cnn_trans(a,b,c)
                print(a,b,c)
                inp,out = cnn.cmipMon(inp_cmip,pr_cmip)
                x_trans,y_trans,x_test,y_test = cnn.transDset(aphro,soda,godas)
                cnn.main(inp,out,x_trans,y_trans,x_test,y_test)
