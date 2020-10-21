import numpy as np
import itertools
import sys
import  threading
import traceback
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
    QLabel, QLCDNumber, QLineEdit, QMainWindow, QMenu, QMessageBox,QTabWidget,QGroupBox,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QStyleFactory,QSpacerItem,QSizePolicy,
    QTextEdit, QToolTip, QVBoxLayout, QWidget, qApp, QWidget)
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import functools
import data_base
import func_base
import mini_base
import sip
import numpy as np

import os

def expand_mask(array,lp=0,rp=0,up=0,dp=0):
    # array is bool
    lp=int(lp)
    rp=int(rp)
    up=int(up)
    dp=int(dp)
    for n in range(lp):
        array = array +np.roll(array,-1-n,axis=1)
    for n in range(rp):
        array = array +np.roll(array,1+n,axis=1)
    for n in range(up):
        array = array +np.roll(array,-1-n,axis=0)
    for n in range(dp):
        array = array +np.roll(array,1+n,axis=0)
    return array

def find_levels(data,threshold,edge=0,larger=False,dura_p=0):
    '''
    edge:
    0: all plat
    1: left edge
    -1: right edge 
    dura: 
    need keep point time

    return :
    mask , index of finds
    '''
    diff=data
    if isinstance(threshold,(list,np.ndarray)):
        threshold=np.array(threshold)
        threshold=threshold[:,None]
        if larger:
            diff=(diff>threshold)*1
        else:
            diff=(diff<threshold)*1
    else:
        if larger:
            diff=(diff>threshold)*1
        else:
            diff=(diff<threshold)*1
    
    #diff=diff-np.roll(diff,edge,axis=1)
    #diff=(diff>0.5)*1
    dura_p=int(dura_p)
    diff_t=diff
    for n in range(dura_p):
        diff_t=diff_t+np.roll(diff,int(-1-n),axis=1)
    diff=(diff_t>dura_p)*1
    if edge:
        return (diff-np.roll(diff,edge,axis=1))>0.5 #,np.where(diff_t>0.5)
    else:
        return (diff>0.5)#, np.where(diff>0.5)

def list_to_dict(labels,values):
    ddict={}
    for x,y in zip(labels,values):
        if x in ddict:
            ddict[x].append(y)
        else:
            ddict[x]=[y]
    return  ddict


def array_vec_thre(array,vector,axis=1,large=True):
    if isinstance(vector,(list,np.ndarray)):
        vector =np.array(vector)
        if axis==1:
            vector=vector[:,None]
        elif axis==0:
            vector=vector[None,:]
    #print(vector)
    #print('finds threshold positive',np.where(array>vector))
    if large:
        return array>vector
    else:
        return array<vector

def tuple_to_dict(ttuplt):
    ddict={}
    for n,p in ttuplt:
        if n in ddict:
            ddict[n].append(p)
        else:
            ddict[n]=[p]
    return ddict


def point_in_mask(mask1,mask2,dura_p):
    length_row=len(mask1[1])
    print('00000length_row',length_row)
    print('00000durap',dura_p)

    l=r=0
    find_levels=[]
    nrow=0
    for row1,row2 in zip(mask1,mask2):
        for cur in range(length_row):
            if (not l) and row1[cur]:
                l=cur
            elif l and (not row1[cur]):
                r=cur
                if r-l> dura_p:
                    #print('bigger')
                    if (row2[l:r]>0).any():
                        #print((l,r))
                        find_levels.append((nrow,l,r-1))
                l=0
        nrow=nrow+1
    return find_levels


def boolean_indexing(v):
    '''
    Convert Python sequence to NumPy array, filling missing values
    equal to a=np.array(list(itertools.zip_longest(*v, fillvalue=0))).T
    '''
    lens = np.array([len(item) for item in v])
    #print(lens)
    mask = lens[:,None] > np.arange(lens.max())
    #print(lens[:,None])
    #print(np.arange(lens.max()))
    out = np.zeros(mask.shape,dtype=int)
    out[mask] = np.concatenate(v)
    return out

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
    
   
def interval_length():
    """
    @param row: 0 1 1-D array
    """
    np.roll()

class BaseWindow(QMainWindow):
    def _create_menu(self, menu_name='',parent_menu=None,StatusTip=None):
        """
        create a primary menu
        @param command:
        @param menu_name:
        @param parent_menu:
        @param text: the menu name
        """

        if parent_menu:
            menu = QMenu(menu_name, self)
            self.menu_dict[menu_name] = menu
            self.menu_dict[parent_menu].addMenu(menu)
        else:
            self.menu_dict[menu_name]=self.menubar.addMenu(menu_name)  # creat 菜单栏中的菜单项
        self.menu_dict[menu_name].setStatusTip(StatusTip)
    def _create_menuact(self,act_name,menu_name,command=None,shortcut=0,StatusTip=None):
        Act = QAction(act_name,self)        # QAction是菜单栏、工具栏或者快捷键的动作的组合
        Act.setShortcut(shortcut)
        Act.setStatusTip(StatusTip)
        if command:
            Act.triggered.connect(command)
        self.menu_dict[menu_name].addAction(Act)
        #print('55',self.menu_dict[menu_name].menuAction())

def seek_files(id1, name):
    """根据输入的文件名称查找对应的文件夹有无改文件，有则输出文件地址"""
    #print(os.walk(id1))
    name=name.strip()
    for root, dirs, files in os.walk(id1):
        #print(files)
        if name in files:
            # 当层文件内有该文件，输出文件地址
            #print(root,name,'*************')
            return os.path.join(root, name)


class EditButtonLine(QWidget):

    def __init__(self, text):
        # QLineEdit.__init__(self,text)
        QWidget.__init__(self)
        self.initUI(text)

    def initUI(self, text):
        self.hbox = QHBoxLayout(self)
        self.initxt = text
        self.line = QLineEdit(text)
        self.button = QPushButton('...')
        self.hbox.addWidget(self.line)
        self.hbox.addWidget(self.button)
        self.button.clicked.connect(self.opendialog)

    def text(self):
        return self.line.text()

    def opendialog(self):
        fname = QFileDialog.getOpenFileName(self, self.text())
        if fname[0]:
            self.line.setText(fname[0])

class FigureCanvasSlot(FigureCanvas):
    def __init__(self,fig):
        #fig = Figure(frameon=False)
        FigureCanvas.__init__(self,fig)
        super().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        super().updateGeometry()
        self.clean()
        #self.canvas = fig.canvas
        self.status=0
        print('creat')
        print(hasattr(self,'mpl_connect'))
        #self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('pick_event', self.one_pick)
        #self.clicked = False

    def on_click(self, event):
        #print(event.__dict__.items())
        print("artist,ind:",event.artist,event.ind)
    #     #if event.inaxes != self.parentax: return
    #     self.mousemotion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
    #     self.clickrelease = self.canvas.mpl_connect('button_release_event', self.on_release)
    #     self.clickx = event.xdata
    #     self.clicky = event.ydata
    #     self.clicked = True
    #     self.temp_artists.clear()

    def clean(self):
        self.temp_inds=set()
        self.temp_artists=set()
        self.artists = set()
        self.inds = set()

    # def on_release(self, event):
    #     self.artists=self.artists.union(self.temp_artists)
    #     self.inds=self.temp_inds.union(self.temp_inds)
    #     self.clicked = False
    #     self.disconnect()
    #
    # def disconnect(self):
    #     self.canvas.mpl_disconnect(self.mousemotion)
    #     self.canvas.mpl_disconnect(self.clickrelease)
    #     self.canvas.draw()
    #
    # def on_motion(self,event):
    #     print(event.artist,event.ind)
    #     # print('\n'.join(['%s:%s' % item for item in event.__dict__.items()]))
    #     self.temp_artists.add(event.artist)
    #     for ind in event.ind:
    #         self.temp_inds.add(ind)
    #
    # def stopdrag(self):
    #     self.myobj.set_url('')
    #     self.canvas.mpl_disconnect(self.clickpress)

    def one_pick(self,event):
        #print('here')
        print('picker:',event.artist,event.ind)
        # print('\n'.join(['%s:%s' % item for item in event.__dict__.items()]))
        self.artists.add(event.artist)
        for ind in event.ind:
            self.inds.add(ind)

    def set_status(self,status):
        """

        @param status: 0:hide 1:show
        """
        #print('set status:',status)
        self.status=status

def abs_distance(coord1,coord2):
    """

    @param coord1:  list like : [(x1,y1),(x2,y2)] as coloumn
    @param coord2:  as row
    """
    arr=np.empty((len(coord2),len(coord1)))
    for row,(a,b) in enumerate(coord2):
        for column,(c,d) in enumerate(coord1):
            ab=abs(a-c)+abs(b-d)
            arr[row,column]=ab
    return  arr


def intersection(coord1,coord2):
    # return coord , index1 , index2
    coord=list( set(coord1) & set(coord2))
    coordict=[[v,0,0] for v in coord]
    index1=[]
    index2=[]
    for n,cor in enumerate(coord1):
        if cor in coord:
            indexx=coord.index(cor)
            coordict[indexx][1]=n
            index1.append(n)
    for n,cor in enumerate(coord2):
        if cor in coord:
            indexx=coord.index(cor)
            coordict[indexx][2]=n
            index2.append(n)
    print('inter',coordict)
    return coordict,index1,index2


def find_index(llist,value1,column1,getcolumn):
    for n,v in enumerate(llist):
        if value1==v[column1]:
            return n,v[getcolumn]



def union_contours(coord1,coord2):
    # if not overlap , return 0
    # if overlap ,return contours
    inters,index1,index2=intersection(coord1,coord2)
    newcontour=[]
    if not  inters:
        return 0
    if coord1[0][0]<coord2[0][0]:
        b0 = coord1[0]
    else:
        # exchange each other
        curcoord=coord2
        coord2=coord1
        coord1=curcoord
        index=index2
        index2=index1
        index1=index
        b0=coord2[0][0]
    len0=len(inters)
    cur1,cur2,cur0=0
    if len0>=len(coord1):
        return coord2
    elif len0>=len(coord2):
        return  coord1

    rout=[]
    cur0=index1[0]
    rout.append((coord1,cur1,cur0))
    while True:
        cur2=find_index(inters,cur0,1,2)
        index0=index2.index(cur2)+1
        if index0>= len(index2):
            break
        cur0=index2[index0]
        rout.append((coord2,cur2,cur0))

        cur1=find_index(inters,cur0,2,1)
        index0=index1.index(cur1)+1
        if index0>=len(index1):
            break
        cur0=index1[index0]
        rout.append((coord1,cur1,cur0))
    for coord , curl,curr in rout:
        newcontour=newcontour+coord[cur1:curr]

    return newcontour



class Mythread(threading.Thread):
    def __init__(self, threadname,func,event):
        threading.Thread.__init__(self, name='线程' + threadname)
        self.threadname = threadname
        self.runflag=True
        self.func=func
        self.event=event
        self.runflag=True
    def run(self):
        #event.wait()
        self.func(self.event)
        self.runflag=False





def decorate_visible(func,name):
    pass

if __name__ == '__main__':
    print(seek_files('I:/Fang/20141211-15d/cs2/VF1','s11-LED2-0Mg_2_MMStack.ome.tif'))
