import re
import threading
import time
import os
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas
# Implement the default Matplotlib key bindings.
from PyQt5 import sip
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
import sys
import PyQt5.QtGui as QtGui
import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.figure import Figure
from PyQt5.QtCore import (QBasicTimer, QCoreApplication, QMimeData, QObject,
                          Qt, pyqtSignal, QSize, QRect)
from PyQt5.QtGui import QColor, QDrag, QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QColorDialog, QComboBox, QDesktopWidget,
    QFileDialog, QFontDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QLCDNumber, QLineEdit, QMainWindow, QMenu, QMessageBox, QTabWidget, QGroupBox,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QStyleFactory, QSpacerItem, QSizePolicy,
    QTextEdit, QToolTip, QVBoxLayout, QWidget, qApp)
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import functools
import mini_base
import func_base
import math
import threading
import  calcium_base
FILEMODES=["txt","txtfolder"]
class Data():
    def __init__(self):
        # TODO : ini some chache space , like dict,list
        pass

    def ini_data_info(self, filename, sep, begin, end, mode=1, sample_freq=10000,filemode=FILEMODES[0]):
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
        self.mode = mode

        self.sep = sep
        self.samplepoints = int((self.end - self.begin) * self.sample_freq + 1)

        self.x_cut_labels = np.arange(self.begin, self.end, 1 / self.sample_freq)
        # data type
        if filemode:
            self.filemode=filemode
        else:
            if os.path.isdir( self.pathway):
                self.filemode=FILEMODES[1]
            else:
                self.filemode=FILEMODES[0]

    def set_chose_drop(self, choses, drops):
        self.choses = choses
        self.drops = drops

    def load_data_info(self):
        if self.filemode==FILEMODES[0]:
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
                # print('err')
                raise IndexError('Unable to recognize separator!!')
        elif self.filemode==FILEMODES[1]:
            print('pathway',self.pathway)
            files=os.listdir(self.pathway)
            self.filelists=[]
            self.sweep_names=[]
            for filename in files:
                sweepnumber = re.sub("\D", "", filename)
                if sweepnumber:
                    sweepnumber = int(sweepnumber)
                    self.sweep_names.append(sweepnumber)
                    self.filelists.append((sweepnumber, filename))
                self.filelists.sort(key=lambda x: x[0])
            print(self.filelists)
            self.filelists=dict(self.filelists)
            if not self.filelists:
                raise IndexError('Don\'t find any sweep in directory !!')

    def load_data(self):
        if self.filemode==FILEMODES[0]:
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
            # self.data_cut = self.data[..., self.sambegin:self.samend]
            # self.cut_duration = len(self.data_cut[1])
            # self.data_cut_index = (np.array(range(self.cut_duration)) +self.begin)

        elif self.filemode==FILEMODES[1]:
            datas = []
            self.choses.sort()
            print(self.choses)
            for sweep in self.choses:
                #print(sweep)
                name=self.filelists[sweep]
                print(os.path.join(self.pathway, name))
                data = pandas.read_csv(os.path.join(self.pathway, name))
                # print("load data - shape :", data.shape)
                datas.append(data.values.T)
            datas = np.array(datas)
            print("load data shape shape", datas.shape)
            datas = np.squeeze(datas, axis=1)
            self.data=datas
        self.data = self.data - np.median(self.data, axis=1, keepdims=True)
        self.total_samplepoints = len(self.data[0])
            #print(filelists)
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
            low_wn4 = 2 * 100 / self.sample_freq
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
        data = self.smooth_data('sosfliter')
        data = data - np.roll(data, 250, axis=1)
        plt.figure()
        sweep_number = len(self.choses)
        bins = np.arange(-30, 30, 1)
        for num, data in enumerate(data, 1):
            ax = plt.subplot(sweep_number + 1, 1, num)
            plt.hist(data, bins=bins)
            a = np.histogram(data, bins=bins)
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
            plt.plot(x_label, ori_wave - n * 50, lw=1, color='gray',label='ori data',pickradius=2)
            #plt.plot(x_label, smooth_wave - n * 50, lw=1, color='lightblue',label='100HZ smooth')
            #plt.plot(x_label, shift_wave - n * 50, lw=1, color='lightgreen',label='100HZ smooth shift')
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
                         label=self.choses[index]
                         ,pickradius=2)
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
                     label="sweep" + str(n + 1)
                     ,pickradius=1)
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

    def CompleteMiniAnalysis(self, method=mini_base.gl_minifind_method[0],dim=5,**kwargs):
        #locals().update(kwargs)
        if method==mini_base.gl_minifind_method[0]:
            data = self.smooth_data('sosfliter')
            # get mini coordinate , regions(minis)
            self.mini_finds, minis = mini_base.minifinds(method,self.data,data, self.sample_freq,
                                                         self.sambegin, self.samend,**kwargs )

            # get mini names : sweep+offset point, need use mini_finds and choses
            mini_names = self.get_mini_names(minis, self.mini_finds)
            n_cluster=kwargs['n_cluster']
            if len(mini_names)<n_cluster:
                print('ERROR: mini number too less < cluster')
                raise IndexError('ERROR: mini number too less < cluster !!')
            # creat mini object
            self.Mini = mini_base.Mini(minis, mini_names,self.mini_finds, self.sample_freq)

            # statistic minis , len(minis),event_sizes,amplitudes,offsets,a_constants,fast_constants,slow_constants
            self.Mini.statis()

            # mini feature extraction  or  Dimensionality reduction
            self.mini_analysis(n_cluster)

    def mini_analysis(self,n_cluster=5):
        if hasattr(self,"mini_result_window"):
            self.mini_result_window.close()
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
        self.make_fig()
        #
        self.mini_result_window = MiniResultWindow(self, 'mini result')
        self.mini_result_window.show()


    def reorgan_mini_finds(self):
        ddict = {}
        for n, sy, ey in self.mini_finds:
            if n in ddict:
                ddict[n].append((sy, ey))
            else:
                ddict[n] = [(sy, ey)]
        return ddict
    def set_root_window(self,widget):
        self.root_window=widget









    def set_color_card(self,n_cluster):
        # set color card
        self.jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=self.Mini.ori_n_cluster - 1)
        self.scalarMap = cm.ScalarMappable(norm=cNorm, cmap=self.jet)
        print('colormap: ',self.scalarMap.get_clim())




    def make_fig(self):
        # --------------------#
        #  three fig     #
        # 1. PCA
        # 2. cluster
        # 3. mini in sweep
        # --------------------#
        self.set_color_card(self.Mini.ori_n_cluster)
        unit = 2
        column = 5
        row = math.ceil(self.Mini.cur_n_cluster / 5)
        wid = column * unit
        hig = row * unit
        if hasattr(self, 'fig1'):
            plt.figure(self.fig1.number)
            plt.clf()  # clear fig
            plt.figure(self.fig2.number)
            plt.clf()
            plt.figure(self.fig3.number)
            plt.clf()
        else:
            self.fig1 = plt.figure(figsize=(5, 5))
            self.fig2 = plt.figure(figsize=(wid, hig))
            self.fig3 = plt.figure(figsize=(8, 12))



        #----  1. PCA fig  ---#

        Xdata = self.Mini.proced_minis[:, 0]
        Ydata = self.Mini.proced_minis[:, 1]
        Zdata = self.Mini.proced_minis[:, 2]

        # draw pca
        plt.figure(self.fig1.number)
        self.fig1ax = plt.subplot(111)
        # For a sequence of values to be color-mapped, use the 'c' argument instead.
        self.fig1scatter=self.fig1ax.scatter(Xdata, Ydata, c=self.Mini.cur_labels, lw=2, cmap=self.jet,pickradius=1)
        self.fig1scatter.set_picker(True)
        # plt.show()

        #----  2. cluster  ---#
        # draw cluster
        t_list = list(self.Mini.cur_labels)
        plt.figure(self.fig2.number)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.gca().patch.set_alpha(0.5)
        # grid = plt.GridSpec(row, column, wspace=0.1, hspace=0.1)
        self.clusteraxs = []
        for n, cluster in enumerate(self.Mini.cur_n_clusters, 1):
            index = t_list.index(cluster)
            mini = self.Mini.minis[index]
            x_label = np.arange(len(mini)) / self.sample_freq
            # print(n//5,n%5,row,column)
            ax = plt.subplot(row, 5, n)
            ax.spines['top'].set_visible(False)  # 去掉上边框
            ax.spines['bottom'].set_visible(False)  # 去掉下边框
            ax.spines['left'].set_visible(False)  # 去掉左边框
            ax.spines['right'].set_visible(False)  # 去掉右边框
            plt.plot(x_label, mini, color=self.scalarMap.to_rgba(self.Mini.cur_labels[index]), pickradius=1)
            self.clusteraxs.append(ax)

        #----  3. mini in sweep  ---#


        #self.mini_in_sweep_ax=plt.subplot(111)
        #self.mini_in_sweep_canvas=self.mini_in_sweep_ax.figure.canvas
        #self.mini_in_sweep_canvas.mpl_connect('button_press_event', self.on_click)
        plt.figure(self.fig3.number)
        reform_mini_finds = self.reorgan_mini_finds()
        self.ax=plt.subplot(111)
        for n, sweep_data in enumerate(self.data):
            plt.plot(self.x_cut_labels, sweep_data[self.sambegin:self.samend] - n * 50, lw=1, color='gray',pickradius=3)
        self.line_minis={}
        for number,((n, p, q) ,mini, label) in enumerate(zip(self.mini_finds, self.Mini.minis, self.Mini.cur_labels)):
            color = self.scalarMap.to_rgba(label)
            x_labels = np.arange(p, q) / self.sample_freq
            # print(len(x_labels))
            # print('mini',len(mini))
            line,=self.ax.plot(x_labels, mini - n * 50, color=color,pickradius=2)
            line.set_picker(True)
            self.line_minis[line]=number






class MiniChoseDialog(QWidget):
    def __init__(self, data,*args, **kwargs):
        print(locals())
        self.data=data
        self.cluster = kwargs['cluster']
        del kwargs['cluster']
        # print(locals())
        super(QWidget, self).__init__(*args, **kwargs)
        self.ini_data(self.cluster)
        self.initUI()
        # self.show()

    def ini_data(self,cluster):
        print((locals()))
        self.data.Mini.reindex_mini()
        self.mini_indexs=self.data.Mini.mini_reindex['label'][cluster]
        #self.mini_indexs = np.where(self.data.Mini.cur_labels == cluster)[0]
        if self.mini_indexs:
            self.cur_index = 0
        else:
            self.cur_index = None
            raise AttributeError('cannot find corespongding minis!!')
        self.mini_number = len(self.mini_indexs)
        self.drops = []

    def initUI(self):
        grid = QGridLayout(self)
        hgrid = QGridLayout()
        hgrid.setGeometry(QRect(120, 22, 100, 400))

        lbutton = QPushButton('<')
        lbutton.setShortcut('Left')
        rbutton = QPushButton('>')
        rbutton.setShortcut('Right')
        self.xcheck = QCheckBox('discard')
        self.all = QCheckBox('discard all')
        self.label = QLabel(
            str(self.cur_index + 1) + '//' + str(self.mini_number))
        self.xcheck.setShortcut('D')
        self.xcheck.stateChanged.connect(self.change_drops)
        lbutton.clicked.connect(self.switch_mini)
        rbutton.clicked.connect(self.switch_mini)

        self.figure = plt.figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)

        self.ax = self.canvas.figure.subplots()
        index = self.mini_indexs[self.cur_index]
        # print('index',index)
        # print('indexs',self.mini_indexs)
        mini_name, mini, label, x_label = self.data.Mini.get_mini_info(index)
        color= self.data.scalarMap.to_rgba(label)
        self.plot, = self.ax.plot(x_label, mini,color=color,pickradius=2)

        self.name=QLabel(mini_name)

        # self.ax.title(mini_name)
        hgrid.addWidget(lbutton, 0, 0)
        hgrid.addWidget(self.canvas, 0, 1, 3, 1)
        hgrid.addWidget(rbutton, 0, 4)

        grid.addWidget(self.all, 0, 0)
        grid.addWidget(self.label, 0, 1)
        grid.addWidget(self.xcheck, 0, 2)
        grid.addWidget(self.name,0,3)
        grid.addLayout(hgrid, 1, 0, 8, 10)

    def change_drops(self):
        button_statues = self.xcheck.checkState() > 0
        drop_status = self.cur_index in self.drops
        if button_statues and not drop_status:
            self.drops.append(self.cur_index)
        if not button_statues and drop_status:
            self.drops.remove(self.cur_index)

    def draw_figure(self, index):
        plt.figure(self.figure.number)
        mini_name, mini, label, x_label = self.data.Mini.get_mini_info(self.mini_indexs[self.cur_index])
        color= self.data.scalarMap.to_rgba(label)

        # print(locals())
        self.ax.set_ylim(min(mini),max(mini))
        self.ax.set_xlim(min(x_label),max(x_label))
        self.plot.set_data(x_label, mini)
        self.plot.figure.canvas.draw()
        self.name.setText(mini_name)

    def switch_mini(self):
        sender = self.sender()  # sender()方法的方式决定了事件源
        message = sender.text()
        if message == '<':
            if self.cur_index > 0:
                self.cur_index = self.cur_index - 1
        elif message == '>':
            if self.cur_index < self.mini_number-1:
                self.cur_index = self.cur_index + 1
        self.draw_figure(self.cur_index)
        self.label.setText(
            str(self.cur_index + 1) + '//' + str(self.mini_number))

        if self.cur_index in self.drops:
            self.xcheck.setChecked(True)
        else:
            self.xcheck.setChecked(False)

    def closeEvent(self, event):
        '''
        if you exit , will excute it
        only for x button,not custom quit button
        '''
        # 消息框,第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里
        reply = QMessageBox.question(
            self, 'Message', "Are you sure to save and quit?",
            QMessageBox.Yes | QMessageBox.Cancel | QMessageBox.No,
            QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.data.Mini.mark_delete_mini(self.drops)
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        elif reply == QMessageBox.Cancel:
            event.ignore()

class MiniResultWindow(func_base.BaseWindow):

    def initUI(self):
        print('iniui')
        self.setGeometry(
            500, 500, 500, 500
        )
        self.ini_bar()
        self.ini_center()
        # self.vbox.addLayout(hbox)
        self.mini_result()
        self.setMouseTracking(True)
        self.show_mini_in_sweep()

    def __init__(self, data,title):
        QWidget.__init__(self)
        self.setWindowTitle(title)
        self.data = data
        self.initUI()
    def ini_center(self):
        self.centerwid=QWidget()
        self.hbox = QHBoxLayout()

        self.centerwid.setLayout(self.hbox)
        self.setCentralWidget(self.centerwid)
    def ini_bar(self):
        self.statusbar = self.statusBar()  # creat 状态栏
        # 调用QtGui.QMainWindow类的statusBar()方法，创建状态栏。第一次调用创建一个状态栏，返回一个状态栏对象。showMessage()方法在状态栏上显示一条信息。
        self.statusbar.showMessage('Ready')
        self.menu_dict = {}
        self.menubar = self.menuBar()  # creat 菜单栏
        # level1 menu
        self._create_menu('&Save')
        self._create_menu('&Load')
        self._create_menu('&View')

        self._create_menu('save mini results', '&Save')
        self._create_menuact('load frame patch txt', '&Load', self.loadframe, 0, 'load calcium frames')
        self._create_menuact('import txt folder', 'import', self.openFolder, 0, 'load multi single sweep')
        self._create_menuact('new', '&File', StatusTip='create sweep')


        self.viewAct=QAction(QIcon('eye.jpg'),'View',self)
        self.viewAct.setStatusTip('show mini')
        self.deleAct=QAction(QIcon('discard.jpg'),'Discard',self)
        self.deleAct.setStatusTip('chose mini you don\'t want')
        self.recoverAct=QAction(QIcon('recovery.jpg'),'Recovery',self)
        self.setStatusTip('recover mini you discarded')
        self.shredderAct=QAction(QIcon('File_Shredder.jpg'),'Completely Discard',self)
        self.shredderAct.setStatusTip('completely delete minis')

        self.viewAct.triggered.connect(lambda : self.set_status(3))
        self.deleAct.triggered.connect(lambda :self.set_status(1))
        self.recoverAct.triggered.connect(lambda : self.set_status(2))

        self.shredderAct.triggered.connect(self.shredder)

        self.toolbar = self.addToolBar('Tool')  # creat 工具栏
        self.toolbar.addActions([self.viewAct,self.deleAct,self.recoverAct,self.shredderAct])

    def loadframe(self):
        self.framfiles=[]
        pathway=os.path.dirname(self.data.pathway)

        fname = QFileDialog.getOpenFileName(self, 'Open tif patch file',
                                            pathway)
        startfile=os.path.join(self.data.pathway,"start time.txt")
        self.starttime=[]
        with open(startfile,'rb',newline='\r\n') as f:
            line=f.readline()
            if "Exposure start      (s): " in line:
                line=line.replace('Exposure start      (s): ',"")
                startime=line.split(',')
                for ti in startime:
                    if ti:
                        self.starttime.append(float(ti))


        # load starttime
        if fname[0]:
            print(fname[0])
            #self.widget.pathway.filename.setText(fname[0])
            self.calciumframe=calcium_base.Calcium(fname,self.data.choses,starttime)

            event=threading.Event()
            caltread=func_base.Mythread('calcium',self.calciumframe.load_all_frame,event)
            caltread.start()
            while (caltread.runflag):
                event.wait()
                self.modifycalcium()
                event.clear()
                time.sleep(2)
            #print('主线程运行时间：%s' % datetime.datetime.now())
            # Flag设置成True
            caltread.join()
    def modifycalcium(self):
        print('finddd!!!~~~')
    def shredder(self):
        #----  remove artist in fugure  ---#
        for artist in self.canvas.discards:
            artist.remove()
            self.canvas.draw()
            self.data.Mini.mark_delete_mini([self.data.line_minis[artist]])
        for artist in self.cluster_label_canvas.arts:
            artist.remove()
            self.canvas.draw()
        self.data.Mini.mark_delete_mini(self.cluster_label_canvas.discards)
        self.data.Mini.truly_delete_mini()
        self.data.mini_analysis()



        #----  remove mini  ---#

    def mini_result(self):
        self.vbox=QVBoxLayout()
        self.cluster_wave_widget(self.vbox) # cluster
        self.cluster_label_show(self.vbox) # PCA
        self.hbox.addLayout((self.vbox))

    def cluster_wave_widget(self,layout):
        # 2. cluster
        if hasattr(self, 'clusters_bks'):
            while self.clusters_bks:
                ck = self.clusters_bks.pop(0)
                self.grid.removeWidget(ck)
                sip.delete(ck)
        print('cluster wave widget')
        self.clusters_bks = []

        self.grid = QGridLayout()
        # show mini cluster : widget buttons
        fig = plt.figure(self.data.fig2.number,figsize=(3,5))

        # return
        print('add cluster frame...')
        for cluster, ax in zip(self.data.Mini.cur_n_clusters, self.data.clusteraxs):
            extent = ax.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
            fig.savefig('target.png', bbox_inches=extent)
            scale = 0.2  # 每次缩小20%
            img = QImage('target.png')  # 创建图片实例
            originWidth = 20
            originHeight = 40
            mgnWidth = int(originWidth * scale)
            mgnHeight = int(originHeight * scale)  # 缩放宽高尺寸
            size = QSize(mgnHeight, mgnWidth)
            pixImg = QPixmap.fromImage(
                img
            )  # .scaled(size, Qt.IgnoreAspectRatio))       #修改图片实例大小并从QImage实例中生成QPixmap实例以备放入QLabel控件中
            bk = QLabel(str(cluster), self)
            bk.resize(mgnWidth, mgnHeight)
            bk.setText(str(cluster))
            bk.setPixmap(pixImg)
            func = functools.partial(self.reject_mini, cluster)
            bk.mouseDoubleClickEvent = func
            self.clusters_bks.append(bk)
            self.grid.addWidget(bk, cluster // 5, cluster % 5)
        layout.addLayout(self.grid)
        # # mini map to data : new figure
        # self.canvas2 = FigureCanvas(plt.figure(DATA.fig.number))
        # self.vbox.addWidget(self.canvas2)
        # plt.show()
    def cluster_label_show(self,layout):
        # 1. PCA
        fig = plt.figure(self.data.fig1.number)
        self.cluster_label_canvas=ScatterFigCanvas(fig,self.data.fig1ax,self.data.fig1scatter)
        layout.addWidget(self.cluster_label_canvas)

    def reject_mini(self, cluster, event):
        # global minichosedialog
        # print(locals())
        self.minichosedialog = MiniChoseDialog(self.data, cluster=cluster)

        self.minichosedialog.show()

    def set_status(self,status):
        self.status=status
        self.canvas.set_status(status)
        self.cluster_label_canvas.set_status(status)



    def show_mini_in_sweep(self):

        # self.mini_maneger=MiniWave()
        # self.mini_maneger.show()

        fig = plt.figure(self.data.fig3.number)
        self.scrollframe = QScrollArea(self)
        self.canvas = PlotFigCanvas(fig)
        #self.canvas.button_press_event()
        self.scrollframe.setWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        # self.button = QPushButton('Plot')
        # self.button.clicked.connect(self.plot)
        vbox = QVBoxLayout()
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.scrollframe)
        self.hbox.addLayout(vbox)
        # self.setLayout(self.vbox)



class ScatterFigCanvas(func_base.FigureCanvasSlot):
    def __init__(self,fig,ax=None,art=None):
        func_base.FigureCanvasSlot.__init__(self,fig)
        #self.artistdict={}
        self.discards=set() # ind
        if ax:
            self.ax=ax
        if art:
            self.art=art
            self.pos = art.get_offsets()
        self.arts=[]
    def one_pick(self,event):
        # print('click', event)
        print(event.ind)
        ind = event.ind[0]
        art=event.artist
        #ind = event.ind
        print(self.art.get_cursor_data(event))
        print(self.status)
        if self.status==1:
            # print('no',self.discards,ind)
            if ind not in self.discards:
                self.discards.add(ind)
                sca=self.ax.scatter([self.pos[ind][0]], [self.pos[ind][1]], c='white',pickradius=1)
                sca.set_picker(True)
                # print(sca)
                self.arts.append(sca)
        elif self.status==2:
            if ind in self.discards:
                self.discards.remove(ind)
            if art in self.arts:
                self.arts.set_visible(False)
        elif self.status==0:
            # show sweep
            pass
        self.draw()

    def mouseeve(self, event):

        print(self.points.contains(event))


class PlotFigCanvas(func_base.FigureCanvasSlot):
    def __init__(self,fig):
        func_base.FigureCanvasSlot.__init__(self,fig)
        self.artistdict={}
        self.discards=set() # art
    # def on_release(self, event):
    #     super().on_release(event)
    #     artists=self.temp_artists
    #     if self.status==1:
    #         for art in artists:
    #             art.set_visible(False)
    #     elif self.status==2:
    #         for art in artists:
    #             art.set_visible(True)
    #     elif self.status==0:
    #         pass
    #     self.canvas.draw()

    def one_pick(self,event):
        #print('onepick,status:',self.status)
        art=event.artist
        #print('pick artist:',art)
        if self.status==1:
            self.discards.add(art)
            art.set_visible(False)
        elif self.status==2:
            art.set_visible(True)
            self.discards.remove(art)
        elif self.status==0:
            # show sweep
            pass
        self.draw()

    # def set_status(self,status):
    #
    #     if status in self.artistdict:
    #         self.artistdict[self.status]=self.artistdict[self.status].union(self.temp_artists)
    #     else:
    #         self.artistdict[self.status]=self.temp_artists
    #
    #     super().set_status(status)

#
# class ScatterFigCanvas(func_base.FigureCanvasSlot):
#     def on_release(self, event):
#         super().on_release(event)
#         artists=self.temp_artists
#         if self.status:
#             for art in artists:
#                 art.set_visible(False)
#         else:
#             for art in artists:
#                 art.set_visible(False)
#         self.draw()
#
#     def one_pick(self,event):
#         art=event.artist
#         if self.status:
#             art.set_visible(False)
#         else:
#             art.set_visible(False)
#         self.draw()