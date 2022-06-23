import os 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class plot_gcm_test():
	def __init__(self,month_n,ldt_n):
		self.month_n = month_n
		self.ldt_n = ldt_n
		self.month_list = ['may','jun','jul','aug','sep','oct']
		self.ldt_list = ['leadtime_1month','leadtime_2month','leadtime_3month']
		self.month = self.month_list[month_n]
		self.ldt = self.ldt_list[ldt_n]
		self.styear = 1993
		self.enyear = 2016
		self.xaxis = np.arange(self.styear,self.enyear)
        self.root = '/docker/mnt/d/research/MtoD/output'+path
	
	def imp(self,path):
		load = lambda x: np.load(os.path.join(self.root,x))
		data = load(path)
		return data
	
	def imp_ecm(self):
		path = 'ECMWF/'+self.ldt+'/bootstrap_'+self.month+'.npy'
		ecm = self.imp(path)
		low,middle,up = ecm[:,0],ecm[:,1],ecm[:,2]
		return low,middle,up
	
	def imp_aphro(self):
		path = 'aphro_month_1951-2015.npy'
		aphro = self.imp(path)
		norm = lambda x: (x-x.mean())/x.std()
		aphro = aphro[self.month_n+4::12]
		aphro = norm(aphro)
		aphro = aphro[-23:]
		return aphro
	
	def imp_gfdl(self):
		path = 'GFDL/gfdl_test.npy'
		gfdl = self.imp(path)
		gfdl = gfdl[self.month_n,self.ldt_n,:]	
		norm = lambda x: (x-x.mean())/x.std() 
		gfdl = norm(gfdl)
		return gfdl
	
	def gcm_test(self):
		low,middle,up = self.imp_ecm()
		gfdl = self.imp_gfdl()
		aphro = self.imp_aphro()
		fig,ax = plt.subplots(figsize=(12,8))
		ax.plot(self.xaxis,middle,marker='o',label='ecmwf')
		ax.fill_between(self.xaxis,low,up,alpha=.5)
		ax.plot(self.xaxis,gfdl,marker='^',label='gfdl')
		ax.plot(self.xaxis,aphro,marker='*',label='aphro')
		plt.legend()
		spath = os.path.join(os.path.dirname(self.root),'pic/gcm_test/' + self.ldt)
		os.makedirs(spath,exist_ok=True)
		sfile = os.path.join(spath,'gcm_test_'+self.month+'.png')
#		plt.savefig(sfile)
		
class plot_gcm_corr():
	def __init__(self):
		self.month_list = ['may','jun','jul','aug','sep','oct']
		self.ldt_list = ['leadtime_1month','leadtime_2month','leadtime_3month']
		self.styear = 1993
		self.enyear = 2016
		self.xaxis = np.arange(self.styear,self.enyear)
        self.root = '/docker/mnt/d/research/MtoD/output'+path
	
	def imp(self):
		load = lambda x: np.load(os.path.join(self.root,x))
		norm = lambda x: (x-x.mean())/x.std()
		path_aphro = 'aphro_month_1951-2015.npy'
		path_gfdl = 'GFDL/gfdl_test_origin.npy'
		aphro = load(path_aphro)
		gfdl = load(path_gfdl)
		aphro_all = []
		gfdl_all = []
		ecm_all = []
		for i in range(len(self.month_list)):
			aphro_month = aphro[i+4::12]
			aphro_norm = norm(aphro_month)
			aphro_all.append(aphro_norm[(self.styear-self.enyear):])
			for j in range(len(self.ldt_list)):
				path_ecm = 'ECMWF/'+self.ldt_list[j]+'/bootstrap_'+self.month_list[i]+'.npy'
				ecm = load(path_ecm)
				ecm_all.append(ecm[:,1])
				gfdl_norm = norm(gfdl[j,i,:])
				gfdl_all.append(gfdl_norm)
		return np.array(aphro_all),ecm_all,gfdl_all
	
	def gcm_corr(self):
		aphro,gfdl,ecm = self.imp()
		aphro_dup = np.array([[x]*3 for x in aphro]).reshape(18,23)
		r_gfdl = []
		r_ecm = []
		for gfdl_id,ecm_id,aphro_id in zip (gfdl,ecm,aphro_dup):
			r_ecm.append(np.corrcoef(gfdl_id,aphro_id)[0,1])
			r_gfdl.append(np.corrcoef(ecm_id,aphro_id)[0,1])
		fig,ax = plt.subplots(figsize=(12,8))
		label_ecm = ['ecmwf_1month','ecmwf_2month','ecmwf_3month']
		label_gfdl = ['gfdl_1month','gfdl_2month','gfdl_3month']
		color_ecm = ['mediumblue','royalblue','lightskyblue']
		color_gfdl = ['mediumvioletred','hotpink','lightpink']
		for i in range(len(self.ldt_list)):
			ax.plot(self.month_list,r_ecm[i::3],marker='o',label=label_ecm[i],color=color_ecm[i])
			ax.plot(self.month_list,r_gfdl[i::3],marker='^',label=label_gfdl[i],color=color_gfdl[i])
		axes = plt.gca()
		axes.set_ylim([0.0,1.0])
		plt.legend()
		spath = os.path.join(os.path.dirname(self.root),'pic/gcm_corr')
		os.makedirs(spath,exist_ok=True)
		sfile = os.path.join(spath,'gcm_corr.png')
		plt.savefig(sfile)

class plot_cnn_cmip():
	def __init__(self):
		self.month_list = ['may','jun','jul','aug','sep','oct']
		self.ldt_list = ['leadtime_1month','leadtime_2month','leadtime_3month']
		self.styear = 1993
		self.enyear = 2016
		self.xaxis = np.arange(self.styear,self.enyear)
        self.root = '/docker/mnt/d/research/MtoD/output'+path
	
	def imp(self):
		load = lambda x: np.load(os.path.join(self.root,x))
		norm = lambda x: (x-x.mean())/x.std()
		row = []
		for i in self.ldt_list:
			column = []
			for j in self.month_list:
				path_dt = 'cnn_cmip/'+i+'/cmip_'+j+'.npy'
				dt = load(path_dt)
				cr = np.mean(dt[0])
				column.append(cr)
			row.append(column)
		return row 

	def cnn_cmip(self):
		heatMap = np.array(self.imp())
		plt.figure(figsize=(15,5))
		plt.imshow(heatMap,interpolation='nearest',cmap='YlOrRd',aspect=0.25,alpha=0.5)
		ys, xs = np.meshgrid(range(heatMap.shape[0]),range(heatMap.shape[1]),indexing='ij')
		for (x,y,val) in zip(xs.flatten(), ys.flatten(), heatMap.flatten()):
			plt.text(x,y,'{0:.2f}'.format(val),horizontalalignment='center',verticalalignment='center',fontsize=20)
		plt.xticks(xs[0,:], self.month_list,fontsize=20)
		plt.yticks(ys[:,0], self.ldt_list,fontsize=20)
		cbar = 	plt.colorbar(aspect=50,orientation = 'horizontal')
		cbar.set_label('correlation skill',size=22)
		cbar.ax.tick_params(labelsize=14)
		plt.tight_layout()
		spath = os.path.join(os.path.dirname(self.root),'pic/cnn_cmip')
		os.makedirs(spath,exist_ok=True)
		sfile = os.path.join(spath,'cnn_cmip_corr.png')
		plt.savefig(sfile)
		plt.show()

class plot_cnn_trans():
	def __init__(self):
		self.month_list = ['may','jun','jul','aug','sep','oct']
		self.ldt_list = ['leadtime_1month','leadtime_2month','leadtime_3month']
		self.styear = 1993
		self.enyear = 2016
		self.xaxis = np.arange(self.styear,self.enyear)
        self.root = '/docker/mnt/d/research/MtoD/output'+path
		self.x = np.arange(1,7)
		self.y = np.arange(-0.1,0.9,0.1)
		self.model_list = ['ECMWF','CNN','GFDL']
		self.hatch_list = ['','x','.']
		
	def imp(self):
		load = lambda x: np.load(os.path.join(self.root,x))
		norm = lambda x: (x-x.mean())/x.std()
		aphro =  load('aphro_month_1951-2015.npy')
		gfdl_low = []
		gfdl_middle = []
		gfdl_up = []
		aphro_all = []
		ecm_low = []
		ecm_middle = []
		ecm_up = []
		cnn_low  = []
		cnn_middle = []
		cnn_up = []
		gfdl_all = []
		for i in range(len(self.month_list)):
			aphro_month = aphro[i+4::12]
			aphro_norm = norm(aphro_month)
			aphro_all.append(aphro_norm[(self.styear-self.enyear):])
			for j in range(len(self.ldt_list)):
				path_gfdl = 'GFDL/'+self.ldt_list[j]+'/bootstrap_'+self.month_list[i]+'.npy'
				path_ecm = 'ECMWF/'+self.ldt_list[j]+'/bootstrap_'+self.month_list[i]+'.npy'
				path_cnn = 'cnn_trans/'+self.ldt_list[j]+'/bootstrap_'+self.month_list[i]+'.npy'
				ecm = load(path_ecm)
				ecm_low.append(ecm[:,0])
				ecm_middle.append(ecm[:,1])
				ecm_up.append(ecm[:,2])
				cnn = load(path_cnn)
				cnn_low.append(cnn[:,0])
				cnn_middle.append(cnn[:,1])
				cnn_up.append(cnn[:,2])
				gfdl = load(path_gfdl)
				gfdl_low.append(gfdl[:,0])
				gfdl_middle.append(gfdl[:,1])
				gfdl_up.append(gfdl[:,2])
		return np.array(aphro_all), ecm_low, ecm_middle, ecm_up, cnn_low, cnn_middle, cnn_up, gfdl_low, gfdl_middle, gfdl_up

	def multibar(self,m):
		aphro,ecm0,ecm1,ecm2,cnn0,cnn1,cnn2,gfdl0,gfdl1,gfdl2 = self.imp()
		aphro_dup = np.array([[x]*3 for x in aphro]).reshape(18,23)
		r_ecm = []
		r_cnn = []
		r_gfdl = []
		for ecm_id,cnn_id,gfdl_id,aphro_id in zip (ecm1,cnn1,gfdl1,aphro_dup):
			r_ecm.append(np.corrcoef(ecm_id,aphro_id)[0,1])
			r_cnn.append(np.corrcoef(cnn_id,aphro_id)[0,1])
			r_gfdl.append(np.corrcoef(gfdl_id,aphro_id)[0,1])
			
		data = [r_ecm[m::3],r_cnn[m::3],r_gfdl[m::3]]
		margin = 0.2
		total_width = 1 -margin

		fig = plt.figure(figsize=(10,5))
		ax = fig.add_subplot(111)
		for i,h in enumerate(data):
			pos = self.x - total_width*(1-(2*i+1)/len(data))/2
			ax.bar(pos, h, width = total_width/len(data),hatch=self.hatch_list[i], alpha=.3, label = self.model_list[i])
		ax.set_xticks(self.x)
		ax.set_xticklabels(self.month_list,fontsize=25)
		ax.set_ylim([0,0.85])
		ax.set_yticks(self.y)
		ax.tick_params(axis='y', labelsize='25')
		ax.set_ylabel('correlation skill',fontsize=25)
		ax.legend(loc=0,prop={'size': 20})
		plt.tight_layout()
		spath = os.path.join(os.path.dirname(self.root),'pic/cnn_trans')
		os.makedirs(spath,exist_ok=True)
		sfile = os.path.join(spath,'multibar'+'_'+self.ldt_list[m]+'.png')
		plt.savefig(sfile)

	def trans_test(self,m,l):
		aphro,ecm0,ecm1,ecm2,cnn0,cnn1,cnn2,gfdl0,gfdl1,gfdl2 = self.imp()
		aphro = aphro[m]
		p = m*3+l
		ecm0,ecm1,ecm2 = ecm0[p],ecm1[p],ecm2[p]
		cnn0,cnn1,cnn2 = cnn0[p],cnn1[p],cnn2[p]
		r_ecm = np.corrcoef(ecm1,aphro)[0,1]
		r_cnn = np.corrcoef(cnn1,aphro)[0,1]
		fig,ax = plt.subplots(figsize=(12,8))
		ax.plot(self.xaxis,ecm1,marker='o',label='ecmwf('+np.str(round(r_ecm,2))+')',color='b')
		ax.fill_between(self.xaxis,ecm0,ecm2,alpha=.3)
		ax.plot(self.xaxis,cnn1,marker='^',label='CNN('+np.str(round(r_cnn,2))+')',color='r')
		ax.fill_between(self.xaxis,cnn0,cnn2,alpha =.3)
		ax.plot(self.xaxis,aphro,marker='s',label='observation',color='k')	
		ax.legend(loc=0, prop={'size':30})
		ax.tick_params(axis='x', labelsize='25')
		ax.tick_params(axis='y', labelsize='25')
		plt.tight_layout()
		spath = os.path.join(os.path.dirname(self.root),'pic/cnn_trans/'+self.ldt_list[l])
		os.makedirs(spath,exist_ok=True)
		sfile = os.path.join(spath,'trans_test_'+self.month_list[m]+'.png')
		plt.savefig(sfile)

	def exec(self):
		for l in range(3):
			self.multibar(l)
			print(f'leadtime is {l}')
			#for m in range(6):
				#self.trans_test(m,l)
				#print(f'leadtime is {l},  month is {m}')
		
if __name__ == '__main__':

	#execute class plot_cnn_trans()
	PLOT = plot_cnn_trans()
	PLOT.exec()

	'''
	#execute class plot_cnn_cmip()
	PLOT = plot_cnn_cmip()
	PLOT.cnn_cmip()
	
	#execute class plot_gcm_corr()
	PLOT = plot_gcm_corr()
	PLOT.gcm_corr()


	#execute class plot_gcm_test() 
	for i in range(6):
		for j in range(3):
			PLOT = plot_gcm_test(i,j)
			PLOT.gcm_test()
			print(i,j)
	'''
