import os
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class aphro():
    def __init__(self):
		self.month = calendar.month_abbr[1:]
		self.resample_size = 50
		self.repeats = 10000
		self.lowlim = 0.025
		self.uplim = 0.975

	def imp(self):
        oppath = '/docker/mnt/d/research/MtoD/output/aphro_month_1951-2015.npy'
		aphro = np.load(oppath)
		aphro = aphro.reshape(65,12).T
		return aphro

	def mkdf(self,d):
		df = pd.DataFrame(d)
		df.index = self.month
		return df
	
	def bts(self,x):
		conf_prob = np.array([self.lowlim,self.uplim])
		re_mean = []
		for i in range(self.repeats):
			re = np.random.choice(x,self.resample_size,replace=True)
			re_mean.append(np.mean(re))
		re_mean_mean = np.mean(re_mean)
		re_mean_std = np.std(re_mean)
		re_conf = np.percentile(re_mean,conf_prob*100)
		return re_mean_mean,re_conf
	
	def exe(self):
		model = 'aphro'
		sfile = []
		APHRO = self.imp()
		DF = self.mkdf(APHRO)	
		for mon in self.month:
			en = DF.loc[mon]
			RE_MEAN_MEAN,RE_CONF = self.bts(en)
			sfile.append([RE_CONF[0],RE_MEAN_MEAN,RE_CONF[1]])
		spath = '/docker/mnt/d/research/MtoD/output/'+model+'/bootstrap_'+model+'.npy'
		np.save(spath,sfile)
		return sfile

	def plot(self,x):
		#low, middle, up = x[0][:],x[1][:],x[2][:]
		low = [x[i][0] for i in range(12)]
		middle = [x[i][1] for i in range(12)]
		up = [x[i][2] for i in range(12)]
		fig,ax = plt.subplots(figsize = (12,8))
		ax.plot(self.month,middle,marker='o')
		ax.fill_between(self.month,low,up,alpha=.5)
		plt.show()

class gfdl():
	def __init__(self):
		self.month_list = ['may','jun','jul','aug','sep','oct']
		self.ldt_list = ['leadtime_1month','leadtime_2month','leadtime_3month']
		self.model = 'GFDL'
		self.resample_size = 10
		self.styear = 1993
		self.enyear = 2015
		self.repeats = 10000
		self.lowlim = 0.025
		self.uplim = 0.975
        self.root = '/docker/mnt/d/research/MtoD/output'
		
	def imp(self,lead,mon):
		oppath = os.path.join(self.root,self.model+'/'+lead)
		load = lambda x: np.load(os.path.join(oppath,x))
		dic = load('en_'+mon+'1993-2015.npy')
		return dic
	
	def mkdf(self,d):
		df = pd.DataFrame(d)
		df.index = np.arange(self.styear,self.enyear+1)
		norm = lambda x: ((x-x.mean())/x.std())
		df_norm = df.apply(norm)
		return df_norm

	def bootstrap(self,x):
		conf_prob = np.array([self.lowlim,self.uplim])
		re_mean = []
		for i in range(self.repeats):
			re = np.random.choice(x,self.resample_size,replace=True)
			re_mean.append(np.mean(re))
		re_mean_mean = np.mean(re_mean)
		re_conf = np.percentile(re_mean,conf_prob*100)
	    return re_mean_mean,re_conf

	def exe(self):
		for mon in self.month_list:
			for lead in self.ldt_list:
				sfile = []
				DIC = self.imp(lead,mon)
				DF_NORM = self.mkdf(DIC)
				for year in range(self.styear,self.enyear+1):
					en = DF_NORM.loc[year]
					RE_MEAN_MEAN,RE_CONF = self.bootstrap(en)
					sfile.append([RE_CONF[0],RE_MEAN_MEAN,RE_CONF[1]])
				spath = os.path.join(self.root,self.model+'/'+lead+'/bootstrap_'+mon+'.npy')
				print(spath)
				np.save(spath,sfile)
		return sfile
				
class bts():
	def __init__(self,target,resample_size):
		self.target = target
		self.styear = 1993
		self.enyear = 2016
		self.resample_size = resample_size
		self.repeats =10000
		self.lowlim = 0.025
		self.uplim = 0.975

	def imp_ecm(self,path):
        oppath = '/docker/mnt/d/research/MtoD/output/ECMWF/'+path
		load = lambda x: np.load(os.path.join(oppath,x))
		dic = load('en_'+self.target+'_1993_2015.npy')
		return dic

	def imp_trans(self,path):
        oppath = '/docker/mnt/d/research/MtoD/output/cnn_trans/'+path
		load = lambda x: np.load(os.path.join(oppath,x))
		trans = load('cmip_'+self.target+'.npy')
		dic = trans.reshape(20,23).T
		return dic

	def imp_trans53(self,path):
        oppath = '/docker/mnt/d/research/MtoD/output/cnn_trans/'+path
		load = lambda x: np.load(os.path.join(oppath,x))
		dic = load('cmip_'+self.target+'.npy')
		dic = dic.reshape(20,22).T
		dic = np.insert(dic,0,0,axis=0)
		print(dic.shape)
		return dic 

	def mkdf(self,d):
		df = pd.DataFrame(d)
		df.index = np.arange(self.styear,self.enyear)
		norm = lambda x:((x-x.mean())/x.std())
		df_norm = df.apply(norm)
		return df_norm

	def bootstrap(self,x):
		conf_prob = np.array([self.lowlim,self.uplim])
		re_mean = []
		for i in range(self.repeats):
			re = np.random.choice(x,self.resample_size,replace=True)
			re_mean.append(np.mean(re))
		re_mean_mean = np.mean(re_mean)
		re_mean_std = np.std(re_mean)
		re_conf = np.percentile(re_mean,conf_prob*100)
		return re_mean_mean,re_conf

if __name__ == '__main__':
	# -------------------------------------------------------------------------------------------------
	#ECMWF: model = 'ECMWF', DIC = BTS.imp_ecm(b), resample_size =20
	#cmip_trans: model = 'cnn_trans', DIC = BTS,imp_trans(b), resample_size =15, if loop for may3lead 
	#--------------------------------------------------------------------------------------------------
	'''
	month_list = ['may','jun','jul','aug','sep','oct']
	ldt_list = ['leadtime_1month','leadtime_2month','leadtime_3month']
	model = 'cnn_trans'
	resample_size = 15
	for a in month_list:
		BTS = bts(a,resample_size)
		for b in ldt_list:
			sfile = []
			if a == 'may'and b == 'leadtime_3month':
				DIC = BTS.imp_trans53(b)
			else:
				DIC = BTS.imp_trans(b)
			DF_NORM = BTS.mkdf(DIC)
			for c in range(1993,2016):
				en = DF_NORM.loc[c]
				RE_MEAN_MEAN,RE_CONF = BTS.bootstrap(en)
				sfile.append([RE_CONF[0],RE_MEAN_MEAN,RE_CONF[1]])
			spath = '/docker/mnt/d/research/MtoD/output/'\
                     +model+'/'+b+'/bootstrap_'+a+'.npy'
			print(spath)
			np.save(spath,sfile)
			
	fig,ax = plt.subplots(figsize=(12,8))
	ax.plot(x,middle,marker='o')
	ax.fill_between(x,low,up,alpha=.5)
	plt.show()
	'''

	#--------------------------------------------------------
	#aphro_bootstrap from jan to Dec 1951-2015
	#--------------------------------------------------------
	'''
	APHRO = aphro()
	SFILE = APHRO.exe()
	APHRO.plot(SFILE)
	'''

	#--------------------------------------------------------
	#GFDL bootstrap 12 ensembles
	#--------------------------------------------------------
	GFDL = gfdl()
	SFILE = GFDL.exe()
