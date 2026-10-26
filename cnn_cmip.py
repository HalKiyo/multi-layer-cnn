import numpy as np
import numpy.ma as ma

from cnn_common import CnnBase, load, targets, is_may_lead3, print_summary

OUTPUT_DIR = '/home/Hasegawa/Dthesis/water/output/cnn_cmip'


class cnn_cmip(CnnBase):
    ensemble = 5

    def train(self, x_train, y_train, x_val, y_val):
        cnn_ensemble = []
        corr_ensemble = []
        rmse_ensemble = []

        for i in range(self.ensemble):

            print('----------------------------------------------------------')
            print(f'Training for ensemble {i} ...')
            # Build a fresh model per member so that the members are independent
            model = self.build_model()

            model.fit(x_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      verbose=2)

            pred = model.predict(x_val)
            corr = np.corrcoef(pred[:, 0], y_val)[0, 1]
            scores = model.evaluate(x_val, y_val, verbose=0)
            rmse = np.sqrt(scores[0])
            print(f'Score for ensemble {i}: RMSE of {rmse}; corr of {corr}')
            cnn_ensemble.append(pred)
            corr_ensemble.append(corr)
            rmse_ensemble.append(rmse)

        print_summary(corr_ensemble, rmse_ensemble)

        outfile = OUTPUT_DIR+'/leadtime_'+str(self.gap-3)+'month/cmip_'+self.sea+'.npy'
        np.save(outfile, [corr_ensemble, rmse_ensemble])


if __name__ == '__main__':
    inp_cmip = load('inp_cmip17.npy')
    pr_cmip = load('pr_cmip17.npy')
    for a, b, c in targets():
        cnn = cnn_cmip(a, b, c)
        print(a, b, c)
        if is_may_lead3(a, c):
            inp, out = cnn.cmipMon53(inp_cmip, pr_cmip)
        else:
            inp, out = cnn.cmipMon(inp_cmip, pr_cmip)
        inp2 = ma.masked_where(inp > 9999, inp)
        x_train, y_train, x_val, y_val = cnn.shuffle(inp2, out)
        cnn.train(x_train, y_train, x_val, y_val)
