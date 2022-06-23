from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers,models,optimizers, losses

class cnn_cmip:
	def __init__(self,m,sea,gap):
		self.m = m
		self.sea = sea
		self.gap = gap
		self.Mon = self.m -self.gap
		self.height,self.width,self.channels = 24,72,6
		self.gcm = 1725
		self.gcm_n = 17
		self.ensemble = 5
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

	def shuffle(self,inputData,outputData):
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

	def checkMask(self,data):
		plt.imshow(data[0,:,:,0],cmap = 'RdBu_r')
		plt.clim(-1.5,1.5)
		plt.colorbar(orientation="horizontal")
		plt.show()

	def build_model(self):
		model = models.Sequential()
		model.add(layers.Conv2D(self.C,(4,8),activation=self.activate_function,input_shape=(self.height,self.width,self.channels),padding='SAME'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(self.C,(2,4),activation = self.activate_function,padding = 'SAME'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(self.C,(2,4),activation = self.activate_function,padding = 'SAME'))
		model.add(layers.Flatten())
		model.add(layers.Dense(self.H, activation=self.activate_function))
		model.add(layers.Dense(1,activation='linear'))

		optimizer = tf.keras.optimizers.RMSprop(self.learnrate)

		model.compile(optimizer=optimizer, loss = self.loss_function, metrics=[self.metrics])
		return model

	def train(self,model,x_train,y_train,x_val,y_val):
		cnn_ensemble = []
		corr_ensemble = []
		rmse_ensemble = []

		for i in range(self.ensemble):

			print('----------------------------------------------------------')
			print(f'Training for ensemble {i} ...')

			history = model.fit(x_train,y_train,
				epochs=self.epochs,
				batch_size=self.batch_size,
				verbose=2)

			pred = model.predict(x_val)
			corr = np.corrcoef(pred[:,0],y_val)[0,1]
			scores = model.evaluate(x_val,y_val,verbose=0)
			rmse = np.sqrt(scores[0])
			print(f'Score for ensemble {i}: RMSE of {rmse}; corr of {corr}')
			cnn_ensemble.append(pred)
			corr_ensemble.append(corr)
			rmse_ensemble.append(np.sqrt(scores[0]))
			
			print('------------------------------------------------------')
			print('Score per ensemble')
			for i in range(0, len(corr_ensemble)):
				print("-----------------------------------------------------")
				print(f'> Fold {i+1} - RMSE: {rmse_ensemble[i]} - Corr: {corr_ensemble[i]}')
			print("-----------------------------------------------------")
			print('Average scores for all folds:')
			print(f'> Corr: {np.mean(corr_ensemble)} (+- {np.std(corr_ensemble)}')
			print(f'> RMSE: {np.mean(rmse_ensemble)} (+- {np.std(rmse_ensemble)}')
			print("-----------------------------------------------------")

		outfile = '/docker/mnt/d/research/MtoD/output/cnn_cmip/leadtime_'\
                   +np.str(self.gap-3)+'month/cmip_'+self.sea+'.npy'
		np.save(outfile,[corr_ensemble,rmse_ensemble])
		
if __name__ == '__main__':
    path = '/docker/mnt/d/research/MtoD/output'
    load = lambda x:np.load(os.path.join(path,x))
    norm = lambda x:((x-x.mean())/x.std())
    inp_cmip = load('inp_cmip17.npy')
    pr_cmip = load('pr_cmip17.npy')
    m_list = np.arange(5,11) # month range from May to Octobe
    sea_list = ['may','jun','jul','aug','sep','oct']
    gap_list = np.arange(4,7) # leadtime-gap for Jan (4=May,5=Jun,6=Jul
    for (a,b) in zip (m_list,sea_list):
        for c in gap_list:
            if a == 5 and c == 6: # prediction of 3 month ahead March requires previous December
                cnn = cnn_cmip(a,b,c)
                print(a,b,c)
                inp,out = cnn.cmipMon53(inp_cmip,pr_cmip)
                inp2 = ma.masked_where(inp>9999,inp)
                x_train,y_train,x_val,y_val = cnn.shuffle(inp2,out)
                model = cnn.build_model()
                cnn.train(model,x_train,y_train,x_val,y_val)
            else:
                cnn = cnn_cmip(a,b,c)
                print(a,b,c)
                inp,out = cnn.cmipMon(inp_cmip,pr_cmip)
                inp2 = ma.masked_where(inp>9999,inp)
                x_train,y_train,x_val,y_val = cnn.shuffle(inp2,out)
                model = cnn.build_model()
                cnn.train(model,x_train,y_train,x_val,y_val)	
