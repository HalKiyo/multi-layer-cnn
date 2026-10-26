import os

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from cnn_common import CnnBase, load, norm, targets, is_may_lead3, print_summary

OUTPUT_DIR = '/home/Hasegawa/Dthesis/water/output/cnn_trans'


class cnn_trans(CnnBase):
    ensemble = 20

    def transDset(self, aphro, soda, godas):
        ap_tr, ap_te = aphro[(self.m-1):42*12], aphro[42*12+(self.m-1):]
        x_trans, y_trans = soda[self.Mon::12, :, :, :], norm(ap_tr[::12])
        x_test, y_test = godas[self.Mon::12], norm(ap_te[::12])
        return x_trans, y_trans, x_test, y_test

    def transDset53(self, aphro, soda, godas):
        ap_tr, ap_te = aphro[(11+5):42*12], aphro[42*12+11+5:]
        x_trans, y_trans = soda[11::12, :, :, :], norm(ap_tr[::12])
        x_test, y_test = godas[11::12], norm(ap_te[::12])
        return x_trans, y_trans, x_test, y_test

    def checkPr(self, data):
        plt.plot(np.arange(len(data)), data)
        plt.ylim()
        plt.show()

    def main(self, inp, out, x_trans, y_trans, x_test, y_test):
        cnn_ensemble = []
        corr_ensemble = []
        rmse_ensemble = []
        # Checkpoints go to ./training_<sea>/ relative to the working directory
        checkpoint_path = 'training_'+self.sea+'/cp.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # The mask does not depend on the ensemble member, so compute it once
        inp2 = ma.masked_where(inp > 9999, inp)
        for i in range(self.ensemble):
            print('----------------------------------------------------')
            print(f'Training for ensemble {i} ...')
            base_model = self.build_model()
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            x_train, y_train, x_val, y_val = self.shuffle(inp2, out)

            base_model.fit(x_train,
                           y_train,
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           validation_data=(x_val, y_val),
                           verbose=1,
                           callbacks=[early_stop, cp_callback]
                           )

            latest = tf.train.latest_checkpoint(checkpoint_dir)

            # Transfer learning. Freeze the convolution and pooling layers 0 to 4
            trans_model = self.build_model()
            trans_model.load_weights(latest)
            for layer in trans_model.layers[:5]:
                layer.trainable = False

            optimizer = tf.keras.optimizers.RMSprop(0.0001)
            trans_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            early_stop2 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)
            trans_model.fit(x_trans, y_trans,
                            epochs=50,
                            validation_data=(x_test, y_test),
                            verbose=1,
                            callbacks=[early_stop2]
                            )

            pred = trans_model.predict(x_test)
            corr = np.corrcoef(pred[:, 0], y_test)[0, 1]
            scores = trans_model.evaluate(x_test, y_test, verbose=0)
            rmse = np.sqrt(scores[0])
            print(f'Score for ensemble {i}: RMSE of {rmse}; corr of {corr}')
            cnn_ensemble.append(pred)
            corr_ensemble.append(corr)
            rmse_ensemble.append(rmse)

        print_summary(corr_ensemble, rmse_ensemble)

        spath = OUTPUT_DIR+'/leadtime_'+str(self.gap-3)+'month'
        os.makedirs(spath, exist_ok=True)
        sfile = os.path.join(spath, 'trans'+self.sea+'.npy')
        # Saving is disabled in the original script. Uncomment to write the predictions.
        # np.save(sfile, cnn_ensemble)
        return sfile


if __name__ == '__main__':
    print('Now Loading ...')
    inp_cmip = load('inp_cmip17.npy')
    pr_cmip = load('pr_cmip17.npy')
    aphro = load('aphro_month_1951-2015.npy')
    godas = load('inp_godas.npy')
    soda = load('soda_train_for_pr.npy')
    soda = ma.masked_where(soda > 9999, soda)
    godas = ma.masked_where(godas > 9999, godas)

    for a, b, c in targets():
        cnn = cnn_trans(a, b, c)
        print(a, b, c)
        if is_may_lead3(a, c):
            inp, out = cnn.cmipMon53(inp_cmip, pr_cmip)
            x_trans, y_trans, x_test, y_test = cnn.transDset53(aphro, soda, godas)
            print(len(x_trans), len(y_trans), len(x_test), len(y_test))
        else:
            inp, out = cnn.cmipMon(inp_cmip, pr_cmip)
            x_trans, y_trans, x_test, y_test = cnn.transDset(aphro, soda, godas)
        cnn.main(inp, out, x_trans, y_trans, x_test, y_test)
