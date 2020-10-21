import math
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from sklearn import datasets
from sklearn.cluster import DBSCAN, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.colors as colors
import matplotlib.cm as cm
import func_base
import mini_base
from types import SimpleNamespace
gl_minifind_method=['pre_cut_off','threshold detect','sosfit']

def minifinds(method,oridata, data, sample_freq, beginp, endp,**kwargs) -> object:
    '''
    data=(np.array) a row is a continues signal
    n_components=5 for dimension of PCA
    data : use for find position
    iridata return the ori mini wave
    '''
    print('-----kwargs------\n',kwargs)
    localvs = SimpleNamespace(**kwargs)
    # rise_time ~1ms
    #print(locals()) # will load  mindura maxdura downthrshold simu_aplitude n_cluster
    pre_n_point=round(0.025*sample_freq)
    # dura_p need small than rise_time
    dura_p=int(0.0005*sample_freq)
    # transform time to point
    # need maintain min duration below -threshold
    if method == gl_minifind_method[0]:
        # -- process data -- #
        #For digital filters, Wn are in the same units as fs.
        # By default, fs is 2 half-cycles/sample,
        # so these are normalized from 0 to 1, where 1 is the Nyquist frequency.
        # (Wn is thus in half-cycles / sample.)
        # For analog filters, Wn is an angular frequency (e.g. rad/s).
        # band pass the data

        # wn =2*pass_freq/sample_freq
        # mini ~ 20ms so 5ms ~ 40ms , 0.005-0.04, 200-25
        # low_wn=2*25/sample_freq
        # hign_wn=2*200/sample_freq
        # sos = signal.butter(8, [low_wn,hign_wn], output='sos',btype="bandpass")
        # data_fliter = signal.sosfiltfilt(sos, data,axis=1)

        # # get baseline need not big current
        # low_wn2=2*0.2/sample_freq
        # sos2 = signal.butter(8, low_wn2, output='sos',btype="lowpass")
        # data_baseline = signal.sosfiltfilt(sos2, data,axis=1)

        # # 50HZ
        # low_wn3=2*45/sample_freq
        # hign_wn3=2*55/sample_freq
        # sos3 = signal.butter(8, [low_wn3,hign_wn3], output='sos',btype="bandpass")
        # data_50hz = signal.sosfiltfilt(sos3, data,axis=1)


        min_dura_p=int(localvs.mindura*sample_freq)
        max_dura_p=int(localvs.maxdura*sample_freq)

        # hign_wn4=2*300/sample_freq
        # sos4 = signal.butter(8, hign_wn4, output='sos',btype="highpass")
        # data_fliter4 = signal.sosfiltfilt(sos4, data,axis=1)

        # data_fliter=signal.savgol_filter(data,13,2,axis=1)
        # x_label=np.arange(0,sample_point)/sample_freq
        # plt.figure()
        # n=0
        # for data1,data2,data3 in zip(data,data_50hz,data_baseline):
        #     plt.plot(x_label,data1+n*50,lw=1,color='gray')
        #     #plt.plot(x_label,data_fliter[0],lw=1,color='green')
        #     plt.plot(x_label,data3+n*50,lw=1,color='red')
        #     plt.plot(x_label,data2+n*50,lw=1,color='orange')
        #     plt.plot(x_label,data4+n*50,lw=1,color='pink')
        #     plt.plot(x_label,data1-data2-data3-data4+n*50,lw=1,color='purple')
        #     n=n+1
        # fli_data =data-data_fliter3-data_fliter2-data_fliter4
        # data =data-data_baseline
        # n=0
        # for data5 in data:
        #     plt.plot(x_label,data5+n*50,lw=1,color='purple')
        #     n=n+1

        # plt.show()
        # return

        sample_point=len(data[0])

        

        # data(x) -data(x-pre_n_point)
        # diff=fli_data-np.roll(fli_data,pre_n_point,axis=1)
        #diff=data-np.roll(data,pre_n_point,axis=1)
        # need omit the first and last pre_n_pint

        #----get rise mask---#
        # get the mini rough offset time need rise maintain 0.5ms
        # stdd=np.std(diff,axis=1)
        # stdd=np.std(diff*(abs(diff)<(3*stdd[:,None])),axis=1)
        # print('diff std',stdd)
        # get mini base fluctuation
        # stds=np.std(data,axis=1)
        # stds=np.std(data*(abs(data)<(downthreshold*stds[:,None])),axis=1)
        stds=np.std(data)
        stds=np.std(data*(abs(data)<(localvs.downthreshold*stds)))
        
        print('data std:',stds)

        mask1=func_base.find_levels(data,-localvs.downthreshold*stds,1,False,dura_p)
        print('mask1 count',len(np.where(mask1)[0]))

        #std_max=max(mini_stds)
        # threshold = 1.5*std_max if 1.5*std_max > threshold else threshold
        # get some stimulus offset time that usually > baseline , and expand lt=0.5s , rt=0.5s
        # get the no mini region upper the threshold
        #mask2 = func_base.array_vec_thre(data,simu_A,axis=1,large=True)
        mask2 = data>localvs.simu_aplitude

        mask2 = func_base.expand_mask(mask2,min_dura_p,min_dura_p)
        mask = mask1 * (~mask2)
        print('mask2 count',len(np.where(mask)[0]))


        # get below -threshold region
        # stdf=np.std(data,axis=1)
        # mask3= func_base.array_vec_thre(data,-stdf,axis=1,large=False)
        # mask3 = func_base.expand_mask(mask3,lp+rp)
        # # exclude suspect stimuls from mini candiate
        # mask=mask*mask3
        mini_finds = np.where(mask)
        #print('mask2 count',len(np.where(mask2)[0]))
        print('mask count',len(mini_finds[0]))



        # reject invalid mini
        t_minis_finds=[]
        flag=True
        row_n = len(data)
        minis=[]
        ey=0
        non=False
        duration_too_long_count=0
        duration_too_short_count=0
        for n in range(row_n): # for each sweep
            ey=tey=0
            starts_index=list(mini_finds[1][mini_finds[0]==n]) 
            for sy in starts_index: # for each sweep minis start time
                large_window=data[n,sy:sy+max_dura_p]
                short_window=data[n,sy-pre_n_point:sy+min_dura_p]
                # out of caculate region
                if sy> endp:
                    break
                # reject multi stack overlap
                if tey>sy:
                    pass
                # reject dacay time less min duration
                elif  np.sum(short_window[-min_dura_p-dura_p:]<(-stds*0.2))<min_dura_p:
                    duration_too_short_count+=1
                #reject duration > 1s :
                elif (large_window<-stds*0.5).all():
                   duration_too_long_count+=1
                   tey=sy+max_dura_p
                # reject out of set range
                # 000011110011
                elif sy<(beginp+min_dura_p):
                    pass
                else:
                    # --find terminate--#
                    # get amplitute
                    amplitude=max(short_window)-min(short_window)
                    # 0.1*amplitute as term 
                    large_window=list(large_window>-0.1*amplitude)
                    try:
                        term=large_window.index(True,min_dura_p)
                    except :
                        print('True is not in list , amplitude = %02f , pos=%d ' %(amplitude,sy))
                    ey=sy+term
                    sy=sy-min_dura_p
                    if None:
                        #if sy-tey<min_dura_p:
                        a,b,c=t_minis_finds[-1]
                        mini_wave = oridata[n, b:ey]
                        minis[-1]=mini_wave
                        t_minis_finds[-1]=((n,b,ey))
                    else:
                        #print(mini_wave)
                        mini_wave = oridata[n, sy:ey]
                        minis.append(mini_wave)
                        t_minis_finds.append((n,sy,ey))
                    if mini_wave.size==0:
                        print('ey sy',ey,sy)
                    tey=ey


        mini_finds=t_minis_finds
        print('begin',pre_n_point+int(beginp))
        print('end',int(endp))
        #print('if1-4',if1,'\t',if2,'\t',if3,'\t',if4,'\t',if5,'\t',if6)
        print('duration too long count:',duration_too_long_count)
        print('duration too short count:',duration_too_short_count)

        #print("mini_finds",mini_finds)
        print("valid count",len(mini_finds))
        #print('mini_finds',[x[1] for x in mini_finds])
        
        return mini_finds,minis


    elif method == 'find_sharp_edge':
        # 做差分 diff1
        # 比较前dura_p个的diff 情况 正的算一起 负的算一起
        # diff_posi=diff*(diff>0)
        # diff_neg=np.abs(diff*(diff<0))
        # for n in range(dura_p):
        #     rolls=np.roll(diff,n+1)
        #     diff_posi=diff_posi+rolls(rolls>0)
        #     diff_neg=diff_neg+np.abs(rolls(rolls<0))
        # mask=func_base.find_levels((diff_neg/diff_posi),14,edge=1,large=True)
        # mini_finds = np.where(mask>0.5)

        # len0=len(mini_finds[0])
        # print('ori count:',len0)
        minis=[]
        mini_finds=[]
        duration_too_long_count=0
        diff=data-np.roll(data,dura_p,axis=1)
        for n,wave in enumerate(diff):
            median=np.median(wave)
            stdd=np.std(wave)
            pre_wave=wave[0:dura_p]
            diff_posi=(pre_wave*(pre_wave>0))**2
            diff_neg=(pre_wave*(pre_wave<0))**2
            pre_wave=list(pre_wave)
            diff_posi=list(diff_posi)
            diff_neg=list(diff_neg)
            sum_posi=sum(diff_posi)
            sum_neg=sum(diff_neg)
            sum_mean=np.mean(pre_wave)
            l=r=0
            retrival=False
            short_dura=int(0.006*sample_freq)
            for cur,value in enumerate(wave[dura_p:],dura_p):
                if sum_posi*10000>sum_neg:
                    radio=sum_neg/sum_posi
                else : 
                    radio=10000
                # using F distribution
                if radio>14 and not l and cur>beginp:
                    #print('rise')
                    l=cur
                elif radio <0.5 and l and (not retrival) :
                    retrival=True
                elif abs(sum_mean)<stdd and retrival :
                    # duration too long > 1s
                    if cur-l > sample_freq:
                        duration_too_long_count=duration_too_long_count+1
                    # duration too small <6ms
                    elif cur<endp and cur-l>short_dura:
                        r=cur+rp
                        minis.append(data[n,l-lp:l+rp])
                        mini_finds.append((n,l))
                    l=0
                    retrival=False
                a=pre_wave.pop(0)
                pre_wave.append(value)
                if a>0:
                    c=diff_posi.pop(0)
                    sum_posi=sum_posi-c
                else:
                    c=diff_neg.pop(0)
                    sum_neg=sum_neg-c
                #if a!=c:
                    #print('error',a,c)
                if value>0:
                    diff_posi.append(value*value)
                    sum_posi=sum_posi+value*value
                else:
                    diff_neg.append(value*value)
                    sum_neg=sum_neg+value*value
                sum_mean=sum_mean+value-a
                
                
        minis=np.array(minis)
        # (x,y) form
        
        print('begin',int(beginp))
        print('end',int(endp))
        #print('if1-4',if1,'\t',if2,'\t',if3,'\t',if4,'\t',if5,'\t',if6)
        print('duration too long count:',duration_too_long_count)

        #print("mini_finds",mini_finds)
        print("valid count",len(mini_finds))
        print('mini_finds',[x[1] for x in mini_finds])
        return mini_finds,minis



class Mini():


    def __init__(self,minis,mini_names,mini_finds,sample_freq):
        self.mini_names =mini_names
        self.minis = minis
        self.sample_freq = sample_freq
        self.mini_finds=mini_finds
        self.offsets= self.fit_paras= self.event_sizes= self.amplitudes= self.fast_constants= self.slow_constants=self.a_constants=self.cur_labels = None
        self.dict=['mini_names','minis','offsets','fit_paras','event_sizes','amplitudes','fast_constants','slow_constants','a_constants','cur_labels','mini_finds']
        self.delete_index = set()

    def _delete_mini(self,index):
        # truly delete
        for name in self.dict:
            if hasattr(self,name):
                llist=getattr(self,name)
                if isinstance(llist,list):
                    llist.pop(index)
                    #print(llist==getattr(self,name))
                else:
                    print(name)
                    setattr(self,name,list(llist))
                    llist = getattr(self, name)
                    llist.pop(index)
    def mark_delete_mini(self,indexs):
        # delete candidate
        # indexs is list or union or tuple
        self.delete_index=self.delete_index.union(indexs)
    def truly_delete_mini(self):
        print(self.delete_index)
        self.delete_index=list(self.delete_index)
        self.delete_index.sort(reverse=True)
        for number in self.delete_index:
            self._delete_mini(number)
        self.delete_index=set() # clear the delete flush

    def reindex_mini(self):
        self.mini_reindex={'label':{},'sweep':{}}
        #self.mini_reindex['label']=func_base.list_to_dict(self.cur_labels,self.minis)
        self.mini_reindex['label']=func_base.list_to_dict(self.cur_labels,range(len(self.cur_labels)))
        #self.mini_reindex['sweep']=func_base.list_to_dict([x[0] for x in self.mini_finds],self.minis)
        self.mini_reindex['sweep']=func_base.list_to_dict([x[0] for x in self.mini_finds],range(len(self.mini_finds)))
        print(self.mini_reindex['label'])
    # self.minis_number,self.event_sizes,self.offsets,self.fast_constants,self.slow_constants,self.rise_10_90s,self.decay_90_50s=mini_base.statis(self.minis)
    def statis(self):
        if not self.minis:
            print('couldn\'t find any minis' )
            return
        #print(self.minis)
        self.mini_number=len(self.minis)


        def templete_func(x,a0,a1,tau1,tau2,t0):

            try:
                return np.piecewise(x,[x>=t0,x<t0],[lambda x: a0+a1*(1-math.exp((x-t0)/tau1))*(math.exp((x-t0)/tau2)),a0])
            except:
                print('xxx',x)

        self.fit_paras=[]
        self.event_sizes=[]
        self.amplitudes=[]
        self.offsets=[]
        self.fast_constants=[]
        self.slow_constants=[]
        self.a_constants=[]


        # fit use two expenent function
        param_bounds=([-np.inf,-np.inf,0,0,-np.inf],[np.inf,0,np.inf,np.inf,np.inf])
        #nn=0
        for mini in self.minis:
            self.amplitudes.append(max(mini)-min(mini))
            minilen= len(mini)
            # if too large  fitcurve cannt work
            if minilen>10000:
                minilen=10000
                mini=mini[:minilen]
            x_label=np.arange(0,minilen)/self.sample_freq
            #nn+=1
            #print(len(x_label))
            try:
                paraments,pcov = curve_fit(templete_func,x_label,mini,bounds=param_bounds)
            except:
                #print(nn)
                print("mini",mini,"label",x_label)
                plt.figure()
                plt.plot(x_label,mini)
                plt.show()
                raise

            self.fit_paras.append(paraments)
            self.offsets.append(paraments[4])
            self.fast_constants.append(paraments[2])
            self.slow_constants.append(paraments[3])
            self.a_constants.append(paraments[1])
            fit_mini=templete_func(x_label,*paraments)
            self.event_sizes.append(max(fit_mini)-min(fit_mini))


    def mini_dim_reduce(self,dim=5):

        # PCA anylysis
        pca=PCA(n_components=dim)
        # Convert Python sequence to NumPy array, filling missing values
        minis=np.array(list(itertools.zip_longest(*self.minis, fillvalue=0))).T
        # transform return array like
        self.proced_minis=pca.fit_transform(minis)

        print('explained variance ratio (first two components): %s' %str(pca.explained_variance_ratio_))

    def get_mini_info(self,index):
        #print(locals())
        mini=self.minis[index]
        x_label=np.arange(len(mini))/self.sample_freq
        return self.mini_names[index],mini,self.cur_labels[index],x_label


    def classify(self,n_cluster=5):
        # Using BIRCH cluster
        self.birch = Birch(threshold=0.5,n_clusters=n_cluster)
        self.birch.fit(self.proced_minis)
        self.ori_labels = self.birch.labels_
        self.ori_centroids = self.birch.subcluster_centers_
        self.ori_n_clusters = np.unique(self.ori_labels)
        self.ori_n_cluster = np.unique(self.ori_labels).size
        self.cur_labels = self.ori_labels
        self.cur_centroids = self.ori_centroids
        self.cur_n_cluster = self.ori_n_cluster
        self.cur_n_clusters = self.ori_n_clusters

    def set_n_cluster(self,n_cluster):
        self.birch.set_params(n_clusters=n_cluster)
        self.cur_labels = self.ori_labels=self.birch.predict(self.proced_minis)
        self.cur_n_cluster = np.unique(self.cur_labels).size
        self.cur_n_clusters = np.unique(self.cur_labels)
        self.cur_centroids = self.birch.subcluster_centers_
    #
    # def make_fig(self):
    #
    #     self.set_color_card(self.ori_n_cluster)
    #     if hasattr(self,'fig1'):
    #         plt.figure(self.fig1.number)
    #         plt.clf()
    #         plt.figure(self.fig2.number)
    #         plt.clf()
    #     else:
    #         self.fig1 = plt.figure(figsize=(5,5))
    #         unit = 2
    #         column = 5
    #         row = math.ceil(self.cur_n_cluster / 5)
    #         wid = column * unit
    #         hig = row * unit
    #         self.fig2=plt.figure( figsize=(wid, hig))
    #
    #     Xdata =self.proced_minis[:,0]
    #     Ydata =self.proced_minis[:,1]
    #     Zdata =self.proced_minis[:,2]
    #
    #     # draw pca
    #     plt.figure(self.fig1.number)
    #     ax=plt.subplot(111)
    #     # For a sequence of values to be color-mapped, use the 'c' argument instead.
    #     ax.scatter(Xdata,Ydata,c=self.cur_labels,lw=2,cmap=self.jet)
    #
    #
    #     #plt.show()
    #
    #     # draw cluster
    #     t_list = list(self.cur_labels)
    #     plt.figure(self.fig2.number)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().patch.set_alpha(0.5)
    #     #grid = plt.GridSpec(row, column, wspace=0.1, hspace=0.1)
    #     self.axs=[]
    #     for n,cluster in enumerate(self.cur_n_clusters,1):
    #         index =t_list.index(cluster)
    #         mini=self.minis[index]
    #         x_label=np.arange(len(mini))/self.sample_freq
    #         #print(n//5,n%5,row,column)
    #         ax=plt.subplot(row,5,n)
    #         ax.spines['top'].set_visible(False) #去掉上边框
    #         ax.spines['bottom'].set_visible(False) #去掉下边框
    #         ax.spines['left'].set_visible(False) #去掉左边框
    #         ax.spines['right'].set_visible(False) #去掉右边框
    #         plt.plot(x_label,mini,color=self.scalarMap.to_rgba(self.cur_labels[index]),pickradius=1)
    #         self.axs.append(ax)
    #
    #
    #
    #
    #
    #
    # def set_color_card(self,n_cluster):
    #     # set color card
    #     self.jet = plt.get_cmap('rainbow')
    #     cNorm = colors.Normalize(vmin=0, vmax=self.ori_n_cluster - 1)
    #     self.scalarMap = cm.ScalarMappable(norm=cNorm, cmap=self.jet)
    #     print('colormap: ',self.scalarMap.get_clim())

    # DBSCAN clustering
    # minis_labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(minis_pca)
    # random_state = 170
    # minis_labels = KMeans(n_clusters=n_cluster,random_state=random_state).fit_predict(minis_pca)
    # n_clusters_ = len(set(minis_labels)) - (1 if -1 in minis_labels else 0)
    # n_noise_ = list(minis_labels).count(-1)
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)

    # mini_finds include offset sweep,points
    # minis include minis waves
    # minis include minis label
    # n_clusters_ = len(set(minis_labels))
    # n_noise_ = list(labels).count(-1)
    # return mini_finds,minis_pca,minis_labels,minis
