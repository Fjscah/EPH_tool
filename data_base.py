import re
import threading
import time

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import medfilt, find_peaks

import mini_base
import func_base


class Data():
    def __init__(self):
        # TODO : ini some chache space , like dict,list
        pass
    def ini_data_info(self,filename,sep,begin,end,mode=1,sample_freq=10000):
        print(locals())
        self.pathway = filename
        # unit : s
        self.begin = float(begin)
        self.end = float(end)
        # unit : HZ
        self.sample_freq = sample_freq

        # unit : point
        self.sambegin = int(self.begin * self.sample_freq)
        self.samend = int(self.sample_freq * self.end)
        self.mode=mode

        self.sep = sep
        self.samplepoints = int((self.end - self.begin) * self.sample_freq + 1)

        self.x_cut_labels = np.arange(self.begin, self.end, 1 / self.sample_freq)
    def set_chose_drop(self, choses, drops):
        self.choses = choses
        self.drops = drops

    def load_data_info(self):
        # load sweep name
        print(self.pathway)
        f = open(self.pathway)
        line_text = f.readline()
        print((line_text))
        if self.sep in line_text:
            # only keep number name
            self.sweep_names = [int(x) for x in re.findall(r'\d+', line_text)]
            return self.sweep_names
        else:
            #print('err')
            raise IndexError('Unable to recognize separator!!')

    def load_data(self):
        self.data = pandas.read_csv(self.pathway, sep=self.sep, na_values='.')
        print("load data - shape :", self.data.shape)
        self.duration = self.data[self.data.columns.values[0]].size
        self.ori_n_sweep = len(self.data.columns.values)
        self.cal_n_sweep = len(self.choses)

        # discard sweeps in drops
        for sweep in self.drops:
            del self.data['sweep' + str(sweep)]
        # print('sweep will be calculate ', self.data.values)
        # unit : s
        self.duration = (self.data[self.data.columns.values[0]].size) / self.sample_freq

        print(self.data.columns.values)

        # print("\nchose", self.chose)
        # print("\ndrop", self.drop)
        # n_sweep = len(self.data.columns.values)

        # transform to array from
        self.data = self.data.values.T
        # print('data type',type(self.data))
        # print(self.data.shape)
        self.data = self.data - np.median(self.data, axis=1, keepdims=True)
        # self.data_cut = self.data[..., self.sambegin:self.samend]
        # self.cut_duration = len(self.data_cut[1])
        # self.data_cut_index = (np.array(range(self.cut_duration)) +self.begin)
        self.total_samplepoints = len(self.data[0])

        # smooth data
        #

    def smooth_data(self, methodname='savgol'):
        '''
        method : savgol , sosfliter, median1,cut_median
        '''

        data = self.data

        print(data.shape)
        if methodname == 'savgol':
            data = signal.savgol_filter(self.data, 7, 2, axis=1)
        elif methodname == 'sosfliter':
            # 100Hz
            low_wn4 = 2 * 400 / self.sample_freq
            sos4 = signal.butter(8, low_wn4, output='sos', btype="lowpass")
            data = signal.sosfiltfilt(sos4, data, axis=1)
            # data = signal.savgol_filter(data,13,2,axis=1)
        elif methodname == 'shrinkage':
            data = data - np.median(data, axis=1, keepdims=True)
            # 0.2HZ 5s get baseline need not big current
            low_wn2 = 2 * 0.2 / self.sample_freq
            sos2 = signal.butter(8, low_wn2, output='sos', btype="lowpass")
            data_baseline = signal.sosfiltfilt(sos2, data, axis=1)
            data = np.subtract(data, data_baseline)
            base_miadans = np.std(data, axis=1).reshape(-1, 1)
            print(base_miadans.T)
            mask1 = np.less(data, 0.0)
            mask2 = np.less(data, -base_miadans)
            find_levels = func_base.point_in_mask(mask1, mask2, int(self.sample_freq * 0.005))
            # print('find_levels',find_levels)
            mask0 = np.zeros(data.shape)
            for x, y1, y2 in find_levels:
                mask0[x, y1:y2] = 1
            mask0 = mask0 > 0.5
            data = data * mask0
        elif methodname == 'findpeak':
            pass
        elif methodname == 'lowpass':
            data = data - np.median(data, axis=1, keepdims=True)
            # 0.2HZ 5s get baseline need not big current
            low_wn2 = 2 * 0.2 / self.sample_freq
            sos2 = signal.butter(8, low_wn2, output='sos', btype="lowpass")
            data_baseline = signal.sosfiltfilt(sos2, data, axis=1)
            data = np.subtract(data, data_baseline)
        elif methodname == 'cut_median':
            pass
        return data

        # data = signal.savgol_filter(self.data,51,5,axis=1)
        # # data=[]
        # # for x in self.data:
        # #     data.append(medfilt(x,int(self.sample_freq/50+1)))
        # # data=np.array(data)

        # print(np.median(data,axis=1).reshape(-1,1))

        sample_freq = self.sample_freq

        # 50HZ
        low_wn3 = 2 * 45 / sample_freq
        hign_wn3 = 2 * 55 / sample_freq
        sos3 = signal.butter(8, [low_wn3, hign_wn3], output='sos', btype="bandpass")
        data_50hz = signal.sosfiltfilt(sos3, data, axis=1)

        data = data - data_50hz - data_baseline  # -data_noise
        # hign_wn4=2*200/sample_freq
        # sos4 = signal.butter(8, hign_wn4, output='sos',btype="highpass")
        # data_noise = signal.sosfiltfilt(sos4, data,axis=1)

        # data=data-data_50hz-data_baseline-data_noise

        return data

    def analysis_FTT(self):
        self.load_data()
        datas = self.data[..., self.sambegin:self.samend]
        sampling_rate = self.sample_freq
        fft_size = self.samplepoints
        t = np.arange(0, 1.0, 1.0 / sampling_rate)
        x = np.sin(2 * np.pi * 20 * t) + 4 * np.sin(2 * np.pi * 10 * t)
        xs = x[:fft_size]
        plt.figure()
        sweep_number = len(self.choses)
        freqs = np.linspace(0, sampling_rate / 2, int(fft_size / 2 + 1))
        plt.ylim(0, 0.5)
        for num, data in enumerate(datas, 1):
            ax = plt.subplot(sweep_number + 1, 1, num)
            ax.set_ylim([0, 0.5])
            xf = np.fft.rfft(data) / fft_size
            xfp = np.abs(xf) * 2
            # print(xf)
            # plt.plot(t,x)
            plt.scatter(freqs[:40000], xfp[:40000])
        plt.show()

    def show_hist(self):
        self.load_data()
        data=self.smooth_data('sosfliter')
        data=data-np.roll(data,250,axis=1)
        plt.figure()
        sweep_number = len(self.choses)
        bins = np.arange(-30, 30, 1)
        for num, data in enumerate(data, 1):
            ax = plt.subplot(sweep_number + 1, 1, num)
            plt.hist(data, bins=bins)
            a=np.histogram(data,bins=bins)
            print(a)
        plt.show()

    def show_sweep(self):

        self.load_data()
        data = self.smooth_data(methodname='sosfliter')
        # data2=self.smooth_data(methodname='lowpass')
        pre_n_point = round(0.025 * self.sample_freq)
        data2 = data - np.roll(data, pre_n_point, axis=1)
        data3 = data - np.roll(data, 1, axis=1)

        x_label = np.arange(0, self.total_samplepoints) / self.sample_freq
        n = 0
        plt.figure()
        plt.title('sweep wave')
        for ori_wave, smooth_wave, shift_wave in zip(self.data, data, data2):
            plt.plot(x_label, ori_wave - n * 50, lw=1, color='gray')
            plt.plot(x_label, smooth_wave - n * 50, lw=1, color='lightblue')
            plt.plot(x_label,shift_wave-n*50,lw=1,color='green')
            # plt.plot(x_label,wave_diff-n*50,lw=1,color='orange')

            n = n + 1
        plt.show()

    def cluster_anylysis(self, n_components=5, random_state=170, n_cluster=5):
        ## PCA anylysis
        self.n_cluster = n_cluster
        pca = PCA(n_components=n_components)
        self.X_r = pca.fit(self.X).transform(self.X)
        explained_variance_ratio_ = pca.explained_variance_ratio_
        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s' %
              str(explained_variance_ratio_))

        ## k mearns anylysis
        # random_state = 170
        self.y_pred = KMeans(n_clusters=n_cluster,
                             random_state=random_state).fit_predict(self.X_r)

    def set_color_card(self):
        # set color card
        self.jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=self.n_cluster - 1)
        self.scalarMap = cm.ScalarMappable(norm=cNorm, cmap=self.jet)
        print(self.scalarMap.get_clim())

    def show_plot(self):
        ## visulation cluster result
        self.set_color_card()

        def check_in_data(posx, posy, X, Y, threshold):
            indexs = None
            distance = threshold
            for x, y, n in zip(X, Y, range(len(X))):
                new_diatance = abs(x - posx) + abs(y - posy)
                if new_diatance < threshold and new_diatance < distance:
                    indexs = n
                    distance = new_diatance
            return indexs

        # x_labels=(np.array(range(duration)))/sample_freq
        def on_click(event):
            if event.inaxes is not None:
                print(event.xdata, event.ydata)
                index = check_in_data(event.xdata, event.ydata, self.X_r[:, 0],
                                      self.X_r[:, 1], 10)
                plt.figure(figsize=((4, 2)))
                plt.title("sweep" + str(index + 1))
                plt.plot(self.x_labels,
                         self.X[index],
                         color=self.scalarMap.to_rgba(self.y_pred[index]),
                         label=self.choses[index])
                plt.show()

            else:
                print('Clicked ouside axes bounds but inside plot window')

        fig = plt.figure()
        lw = 2
        plt.scatter(
            self.X_r[:, 0],
            self.X_r[:, 1],
            c=self.y_pred,
            lw=lw,
            cmap=self.jet,
        )

        fig.canvas.callbacks.connect('button_press_event', on_click)
        plt.title('PCA + k-means of sweep ?-?' + str(self.x_labels[0]) + '-' +
                  str(self.x_labels[-1]) + 's dataset')
        # plt.show()

        # show all wave
        fig = plt.figure()
        for n, single_wave in enumerate(self.X):
            single_wave = (np.array(single_wave) - min(single_wave)) / (
                    max(single_wave) - min(single_wave))
            plt.plot(self.x_labels,
                     single_wave - 2 * n,
                     color=self.scalarMap.to_rgba(self.y_pred[n]),
                     label="sweep" + str(n + 1))
        plt.title('PCA + k-means of sweep ?-?  ' + str(self.x_labels[0]) +
                  '-' + str(self.x_labels[-1]) + 's dataset')
        plt.show()

        # show all wave
        fig = plt.figure()
        for n, single_wave in enumerate(self.X):
            single_wave = np.array(single_wave)
            plt.plot(self.x_labels,
                     single_wave - 50 * n,
                     color=self.scalarMap.to_rgba(self.y_pred[n]),
                     label="sweep" + str(n + 1))
        plt.title('PCA + k-means of sweep 1-26  ' + str(self.x_labels[0]) +
                  '-' + str(self.x_labels[-1]) + 's dataset')
        plt.show()

    def get_mini_names(self, minis, mini_finds):
        mini_names = []
        for mini, cor in zip(minis, mini_finds):
            mini_names.append(
                "mini_" + str(self.choses[cor[0]]) + "_" +
                str(int((cor[1] / 1000 * self.sample_freq))) + "_" +
                str(int((cor[2] / 1000 * self.sample_freq))))
        return mini_names

    def CompleteMiniAnalysis(self, downthreshold=1.5, simu_aplitude=3, lt=0.01, rt=0.02, n_cluster=4, dim=5):

        data = self.smooth_data('sosfliter')
        # get mini coordinate , regions(minis)  
        self.mini_finds, minis = mini_base.minifinds(self.data,
                                                     data, self.sample_freq, downthreshold, simu_aplitude, lt, rt,
                                                     self.sambegin, self.samend, method='pre_cut_off')

        # get mini names : sweep+offset point, need use mini_finds and choses
        mini_names = self.get_mini_names(minis, self.mini_finds)

        # creat mini object
        self.Mini = mini_base.Mini(minis, mini_names, self.sample_freq)

        # statistic minis , len(minis),event_sizes,amplitudes,offsets,a_constants,fast_constants,slow_constants
        #self.Mini.statis()

        # mini feature extraction  or  Dimensionality reduction
        self.Mini.mini_dim_reduce(dim=5)

        # --- CLUSTER --- #
        # get mini clasification labels, if has -1 stand for noise (what's noise ? maybe outliers)
        self.Mini.classify(n_cluster)
        self.Mini.set_n_cluster(n_cluster)

        # get mini classification cluster
        # self.mini_n_clusters=list(set(self.minis_labels))
        # self.mini_n_cluster =len(self.mini_n_clusters)
        # if -1 in self.mini_n_clusters:
        #     self.mini_noise = True
        #     self.mini_n_cluster= self.mini_n_cluster-1
        # else:
        #     self.mini_noise = False
        # --- CLUSTER --- #

        # store in fig form , wait for show
        self.Mini.make_fig()
        self.make_fig()

    def reorgan_mini_finds(self):
        ddict = {}
        for n, sy, ey in self.mini_finds:
            if n in ddict:
                ddict[n].append((sy, ey))
            else:
                ddict[n] = [(sy, ey)]
        return ddict

    def make_fig(self):
        self.mini_in_sweep_fig = plt.figure()
        plt.subplot(111)
        reform_mini_finds = self.reorgan_mini_finds()
        for n, sweep_data in enumerate(self.data):
            plt.plot(self.x_cut_labels, sweep_data[self.sambegin:self.samend] - n * 50, lw=1, color='gray')
        for (n, p, q), mini, label in zip(self.mini_finds, self.Mini.minis, self.Mini.cur_labels):
            color = self.Mini.scalarMap.to_rgba(label)
            x_labels = np.arange(p, q) / self.sample_freq
            # print(len(x_labels))
            # print('mini',len(mini))
            plt.plot(x_labels, mini - n * 50, color=color)
