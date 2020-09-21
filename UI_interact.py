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
    QLabel, QLCDNumber, QLineEdit, QMainWindow, QMenu, QMessageBox,QTabWidget,QGroupBox,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QStyleFactory,QSpacerItem,QSizePolicy,
    QTextEdit, QToolTip, QVBoxLayout, QWidget, qApp)
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import functools
import data_base
import func_base
import mini_base
import sip
import numpy as np

if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (FigureCanvas,
                                                    NavigationToolbar2QT as
                                                    NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (FigureCanvas,
                                                    NavigationToolbar2QT as
                                                    NavigationToolbar)

gl_DATA = data_base.Data()
minichosedialog = None


class Communicate(QObject):
    closeApp = pyqtSignal()  # Communicate类创建了一个pyqtSignal()属性的信号。

    def condition_exit(self, key):
        if key == 'q':
            self.c.closeApp.emit()

    def create_DATA(self):
        # TODO : maybe need change global DATA to here
        pass


class Application(QMainWindow):
    # 主窗口 , QMainWindow提供了主窗口的功能，使用它能创建一些简单的状态栏、工具栏和菜单栏。class QMainWindow(QWidget)
    def __init__(self):
        super().__init__()  # use qwidget to creat window
        self.initUI()

    def initUI(self):
        """
        initial the main window layout , and add analysis tool
        """
        # window.resize(250, 150)
        # window.move(500, 500) # move window in scrrean coorsinate
        # setGeometry()有两个作用：把窗口放到屏幕上并且设置窗口大小。参数分别代表屏幕坐标的x、y和窗口大小的宽、高。也就是说这个方法是resize()和move()的合体
        self.setGeometry(
            500, 500, 500, 500
        )
        # set window title
        self.setWindowTitle('electrophysiology analysis')
        # set window icon
        self.setWindowIcon(QIcon('user.png'))
        # set the main window plane which add various analysis tools
        self.widget = CenterWidget()
        self.setCentralWidget(self.widget)

        # set menu and state bar
        self.ini_bar()

        # put window in center
        self.center()
        self.setMouseTracking(True)  # 事件追踪默认没有开启，当开启后才会追踪鼠标的点击事件

        self.c = Communicate()
        self.c.closeApp.connect(self.close)
        # self.ini_button()
        # self.tip_ui()
        # self.quit_ui()
        # self.ini_label
        # textEdit = QTextEdit()
        # self.setCentralWidget(textEdit)#Sets the given widget to be the main window's central widget.

        self.show()  # IMPORTANT!!!!! Windows are hidden by default.

    def ini_bar(self):
        '''
        include : status, menu, tool bar
        '''

        self.statusbar = self.statusBar()  # creat 状态栏
        # 调用QtGui.QMainWindow类的statusBar()方法，创建状态栏。第一次调用创建一个状态栏，返回一个状态栏对象。showMessage()方法在状态栏上显示一条信息。
        self.statusbar.showMessage('Ready')
        self.menu_dict = {}
        self.menubar = self.menuBar()  # creat 菜单栏
        # level1 menu
        self._create_menu('&File')
        self._create_menu('&Analysis')

        # level2 menu and menuaction
        # add &file
        self._create_menuact('exit','&File',qApp.quit,'Ctrl+Q','Exit application')
        self._create_menu('import','&File')
        self._create_menuact('import txt','import',self.openFile,0,'load txt sweep')
        self._create_menuact('new','&File',StatusTip='create sweep')

        self._create_menuact('CompleteMiniAnalysis','&Analysis',self.widget.mini_plane,0,StatusTip='analysis minis')
        self._create_menuact('Sweep Waves','&Analysis',self.show_sweep,0,StatusTip='show sweeps waves(smooth)')
        self._create_menuact('FTT frequency spectrum','&Analysis',self.show_FTT,0,StatusTip='analysis FTT')
        self._create_menuact('HIST','&Analysis',self.show_hist,0,StatusTip='analysis hist')


        # self.toolbar = self.addToolBar('Exit')  # creat 工具栏
        # self.toolbar.addAction(exitAct)
    def _create_menu(self, menu_name,parent_menu=None,StatusTip=None):
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

    def show_sweep(self):
        global gl_DATA
        gl_DATA.show_sweep()

    def show_FTT(self):
        global gl_DATA
        gl_DATA.analysis_FTT()
    def show_hist(self):
        global gl_DATA
        gl_DATA.show_hist()
    def center(self):
        '''
        put widget in the center of creen
        '''
        geom = self.frameGeometry()  # 获得主窗口所在的框架
        screen_geom = QDesktopWidget().availableGeometry().center(
        )  # 获取显示器的分辨率，然后得到屏幕中间点的位置
        geom.moveCenter(screen_geom)  # 然后把主窗口框架的中心点放置到屏幕的中心位置
        self.move(geom.topLeft())  # 然后通过move函数把主窗口的左上角移动到其框架的左上角，这样就把窗口居中了

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'F:/DATA/minis/')
        if fname[0]:
            self.widget.pathway.filename.setText(fname[0])

    def closeEvent(self, event):
        '''
        if you exit , will excute it
        only for x button,not custom quit button
        '''
        # 消息框,第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            plt.close('all')
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, e):
        self.setMouseTracking(True)
        x = e.x()
        y = e.y()
        # if x>10:
        #     self.c.condition_exit('q')
        text = "x:{0} , y:{1}".format(x, y)
        # print(text)
        self.statusbar.showMessage(text)

    def show_mouse_pos(self):
        grid = QGridLayout()
        grid.setSpacing(10)

        x = 0
        y = 0
        text = "x:{0} , y:{1}".format(x, y)

        self.label = QLabel(self.text, self)
        grid.addWidget(self.label, 0, 0, Qt.AlignBottom)

        self.setMouseTracking(True)
        self.setLayout(grid)

    def keyPressEvent(self, e):
        '''
        detect key press event
        '''
        # 替换了事件处理器函数keyPressEvent(),如果按下ESC键程序就会退出
        if e.key() == Qt.Key_Escape:
            self.close()

    def _toggleMenu(self, state):
        # toggle : status switch
        if state:
            self.statusbar.show()
        else:
            self.statusbar.hide()

    def contextMenuEvent(self, event):
        '''
        right click menu
        右键菜单也叫弹出框（！？），是在某些场合下显示的一组命令。
        例如，Opera浏览器里，网页上的右键菜单里会有刷新，返回或者查看页面源代码。
        如果在工具栏上右键，会得到一个不同的用来管理工具栏的菜单。
        '''
        cmenu = QMenu(self)
        newAct = cmenu.addAction("New")
        openAct = cmenu.addAction("Open")
        quiAct = cmenu.addAction("Quit")
        action = cmenu.exec_(
            self.mapToGlobal(event.pos())
        )  # 使用exec_()方法显示菜单。从鼠标右键事件对象中获得当前坐标。mapToGlobal()方法把当前组件的相对坐标转换为窗口（window）的绝对坐标,返回选中结果
        if action == quiAct:
            qApp.quit()

    def ini_button(self):
        btn1 = QPushButton("Button 1", self)
        btn1.move(30, 50)

        btn2 = QPushButton("Button 2", self)
        btn2.move(150, 50)
        # 两个按钮都和同一个slot绑定
        btn1.clicked.connect(self.buttonClicked)
        btn2.clicked.connect(self.buttonClicked)

        self.statusBar()

        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('Event sender')
        self.show()

    def buttonClicked(self):

        sender = self.sender()  # sender()方法的方式决定了事件源
        self.statusBar().showMessage(sender.text() + ' was pressed')


# class ImgQtFigButton(QPushButton):
#     def __init__(self,data,color='blue'):
#         plt.figure(figsize=(0.2,0.2),facecolor="#FFDAB9")
#         ax=plt.subplot(111,)
#         ax.spines['top'].set_visible(False) #去掉上边框
#         ax.spines['bottom'].set_visible(False) #去掉下边框
#         ax.spines['left'].set_visible(False) #去掉左边框
#         ax.spines['right'].set_visible(False) #去掉右边框
#         plt.plot(data,linewidth=0.5,color=color)

#         fig = plt.gcf()

#         #plt.imshow(***[等待保存的图片]***)
#         #fig.pixels(100,100)
#         #fig.set_size_inches(19.2/3,10.8/3) #dpi = 300, output = 700*700 pixels
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         plt.gca().patch.set_alpha(0.5)
#         plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#         plt.margins(0,0)
#         fig.savefig('target.png', format='png', transparent=True, dpi=300, pad_inches = 0)
#         plt.clf()
#         scale = 0.8		#每次缩小20%
#         img = QImage('target.png')    #创建图片实例
#         originHeight=50
#         originWidth=70
#         mgnWidth = int(originWidth * scale)
#         mgnHeight = int(originHeight * scale)    #缩放宽高尺寸
#         size = QSize(mgnWidth, mgnHeight)

#         pixImg = QPixmap.fromImage(img.scaled(size, Qt.IgnoreAspectRatio))       #修改图片实例大小并从QImage实例中生成QPixmap实例以备放入QLabel控件中

#         self.imageLabel.resize(mgnWidth, mgnHeight)
#         self.imageLabel.setPixmap(pixImg)
#         self.image= ImageTk.PhotoImage(file ='target.png' )


class CenterWidget(QWidget):
    def __init__(self):
        super().__init__()  # use qwidget to creat window
        self.initUI()

    def initUI(self):

        self.setMouseTracking(True)

        self.pathway = PathDialog()

        self.vbox = QVBoxLayout(self)
        self.vbox.addWidget(self.pathway)

        self.setLayout(self.vbox)
        self.setGeometry(100,100, 100, 40)
        self.setWindowTitle('hahahaha')

        self.setAcceptDrops(True)  # 允许拖拽
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vbox.addItem(verticalSpacer)

    def mini_plane(self):
        global gl_DATA
        if hasattr(self, 'miniplane'):
            if self.miniplane.isVisible():
                self.miniplane.setVisible(False)
            else:
                self.miniplane.setVisible(True)

        else:
            self.miniplane = MiniPlane()
            self.vbox.insertWidget(1,self.miniplane)


class MiniPlane(QGroupBox):
    def __init__(self):
        QGroupBox.__init__(self,'Mini anylysis')
        self.initUI()

    def initUI(self):
        ltlable = QLabel('left time (s)')
        rtlabel = QLabel('right time(s)')
        duralabel=QLabel('min duration time(s)')

        threshold = QLabel('threshold(std)')
        simu_amplitude=QLabel('simu_intensity')
        clusterlabel = QLabel('cluster number')

        self.ltname = QLineEdit('0.01')
        self.rtname = QLineEdit('0.02')
        self.minduname=QLineEdit('0.005')
        self.downthresholdname = QLineEdit('4')
        self.simuname=QLineEdit('50')

        self.clustername = QLineEdit('5')

        hbox = QHBoxLayout()
        hbox.addWidget(ltlable)
        hbox.addWidget(self.ltname)
        hbox.addWidget(rtlabel)
        hbox.addWidget(self.rtname)
        hbox.addWidget(duralabel)
        hbox.addWidget(self.minduname)
        hbox.addWidget(threshold)
        hbox.addWidget(self.downthresholdname)
        hbox.addWidget(simu_amplitude)
        hbox.addWidget(self.simuname)

        hbox.addWidget(clusterlabel)
        hbox.addWidget(self.clustername)

        self.ok = QPushButton('OK')
        hbox.addWidget(self.ok)
        self.ok.clicked.connect(self.analysis)

        self.vbox = QVBoxLayout(self)
        self.vbox.addLayout(hbox)

    def analysis(self):
        global gl_DATA
        # threshold=10,lt=0.01,rt=0.02,n_cluster=4,dim=5
        gl_DATA.load_data()
        gl_DATA.CompleteMiniAnalysis(float(self.downthresholdname.text()),
                                     float(self.simuname.text()),
                                     float(self.minduname.text()),
                                     float(self.ltname.text()),
                                     int(self.clustername.text()))
        # print(DATA.Mini.fig1.number)
        # PCA result : new figure
        # self.canvas1 = FigureCanvas(plt.figure(DATA.Mini.fig1.number))
        # self.vbox.addWidget(self.canvas1)
        if hasattr(self, 'clusters_bks'):
            while self.clusters_bks:
                ck = self.clusters_bks.pop(0)
                self.grid.removeWidget(ck)
                sip.delete(ck)

        self.clusters_bks = []

        self.grid = QGridLayout()
        # show mini cluster : widget buttons
        fig = plt.figure(gl_DATA.Mini.fig2.number)
        # return
        for cluster, ax in zip(gl_DATA.Mini.cur_n_clusters, gl_DATA.Mini.axs):
            # print('add')
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
        self.vbox.addLayout(self.grid)
        self.show_mini_in_sweep()
        # # mini map to data : new figure
        # self.canvas2 = FigureCanvas(plt.figure(DATA.fig.number))
        # self.vbox.addWidget(self.canvas2)
        plt.show()

    def show_mini_in_sweep(self):

        self.mini_maneger=MiniWave()
        self.mini_maneger.show()


    def reject_mini(self, cluster, event):
        global minichosedialog
        # print(locals())
        # print('here')
        minichosedialog = MiniChoseDialog(cluster=cluster)
        minichosedialog.show()

class MiniWave(QWidget):
    def __init__(self, *args, **kwargs):
        # print(locals())
        super(QWidget, self).__init__(*args, **kwargs)
        self.initUI()
        # self.show()

    def initUI(self):
        global gl_DATA
        fig = plt.figure(gl_DATA.mini_in_sweep_fig.number)
        self.scrollframe=QScrollArea(self)
        self.canvas=FigureCanvas(fig)
        self.scrollframe.setWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        #self.button = QPushButton('Plot')
        #self.button.clicked.connect(self.plot)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.scrollframe)
        self.setLayout(self.vbox)


class MiniChoseDialog(QWidget):
    def __init__(self, *args, **kwargs):
        # print(locals())
        self.cluster = kwargs['cluster']
        del kwargs['cluster']
        # print(locals())
        super(QWidget, self).__init__(*args, **kwargs)
        self.ini_data(self.cluster)
        self.initUI()
        # self.show()

    def ini_data(self, cluster):
        global gl_DATA
        self.mini_indexs = np.where(gl_DATA.Mini.cur_labels == cluster)[0]
        if self.mini_indexs.any():
            self.cur_index = 0
        else:
            self.cur_index = None
        self.mini_number = len(self.mini_indexs)
        self.drops = []

    def initUI(self):
        grid = QGridLayout(self)
        hgrid=QGridLayout()
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
        mini_name, mini, label, color, x_label = gl_DATA.Mini.get_mini_info(index)
        self.plot, = self.canvas.figure.subplots().plot(x_label,
                                                        mini,
                                                        color=color)

        # self.ax.title(mini_name)
        hgrid.addWidget(lbutton,0,0)
        hgrid.addWidget(self.canvas,0,1,3,1 )
        hgrid.addWidget(rbutton,0,4)

        grid.addWidget(self.all, 0, 0)
        grid.addWidget(self.label, 0, 1)
        grid.addWidget(self.xcheck, 0, 2)
        grid.addLayout(hgrid, 1, 0, 4, 6)

    def change_drops(self):
        button_statues = self.xcheck.checkState() > 0
        drop_status = self.cur_index in self.drops
        if button_statues and not drop_status:
            self.drops.append(self.cur_index)
        if not button_statues and drop_status:
            self.drops.remove(self.cur_index)

    def draw_figure(self, index):
        global gl_DATA
        plt.figure(self.figure.number)
        mini_name, mini, label, color, x_label = gl_DATA.Mini.get_mini_info(index)
        # print(locals())
        self.plot.set_data(x_label, mini)
        self.plot.figure.canvas.draw()

    def switch_mini(self):
        sender = self.sender()  # sender()方法的方式决定了事件源
        message = sender.text()
        if message == '<':
            if self.cur_index > 0:
                self.cur_index = self.cur_index - 1
        elif message == '>':
            if self.cur_index < self.mini_number:
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
            global gl_DATA
            for index in self.drops:
                gl_DATA.Mini.delete_mini(index)

            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        elif reply == QMessageBox.Cancel:
            event.ignore()


class PathDialog(QGroupBox):
    def __init__(self):
        QGroupBox.__init__(self,'file information setting')
        self.initUI()
        #self.setBaseSize()

    def initUI(self):

        hbox1 = QHBoxLayout()

        filelabel = QLabel('pathway:')
        self.filename = QLineEdit(
            'F:\\project\\python\\simulation sweep data\\new.csv')
        seplabel = QLabel('sep')
        self.sepname = QLineEdit('\t')
        modelabel=QLabel('mode')
        self.modename=QComboBox()

        self.ok = QPushButton('OK')
        self.ok.clicked.connect(self.ini_communicaton)

        self.modename.addItem('current',1)
        self.modename.addItem('voltage',2)

        hbox1.addWidget(filelabel)
        hbox1.addWidget(self.filename)
        hbox1.addWidget(seplabel)
        hbox1.addWidget(self.sepname)
        hbox1.addWidget(modelabel)
        hbox1.addWidget((self.modename))
        hbox1.addWidget(self.ok)

        hbox2 = QHBoxLayout()
        beginlabel = QLabel('begin(s):')
        endlabel = QLabel('end(s):')
        samplelabel = QLabel('sample frequency(HZ):')
        self.beginname = QLineEdit('4')
        self.endname = QLineEdit('30')
        self.samplename = QLineEdit('10000')
        hbox2.addWidget(beginlabel)
        hbox2.addWidget(self.beginname)
        hbox2.addWidget(endlabel)
        hbox2.addWidget(self.endname)
        hbox2.addWidget(samplelabel)
        hbox2.addWidget(self.samplename)

        self.vbox = QVBoxLayout(self)
        self.vbox.addLayout(hbox1)
        self.vbox.addLayout(hbox2)

    def ini_communicaton(self):
        global gl_DATA
        print('combo',self.modename.currentData())
        gl_DATA.ini_data_info(self.filename.text(), self.sepname.text(),
                                 float(self.beginname.text()),
                                 float(self.endname.text()),
                              int(self.modename.currentData()),
                                 int(self.samplename.text()),
                              )
        try:
            gl_DATA.load_data_info()
            if hasattr(self, 'cks'):
                while self.cks:
                    ck = self.cks.pop(0)
                    self.grid.removeWidget(ck)
                    sip.delete(ck)
            self.cks = []
            self.grid = QGridLayout()
            self.grid.setSpacing(10)
            for n, number in enumerate(gl_DATA.sweep_names):
                cb = QCheckBox(str(number))
                cb.setChecked(True)
                cb.stateChanged.connect(self.set_drops_choses)
                self.grid.addWidget(cb, 1 + n // 10, n % 10, Qt.AlignBottom)
                self.cks.append(cb)
            gl_DATA.set_chose_drop(gl_DATA.sweep_names, [])
            self.vbox.addLayout(self.grid)
            # self.set_drops_choses()
            if hasattr(self, 'all'):
                pass
            else:
                self.all = QCheckBox('all')
            self.grid.addWidget(self.all, 0, 0)
            self.all.stateChanged.connect(self.set_all_choses)
        except IndexError as e:
            #print('ee')
            QMessageBox.information(self, "ERROR Information",self.tr(str(e)))

    def set_all_choses(self):
        status=self.all.checkState()
        for ck in self.cks:
            ck.setCheckState(status)

    def set_drops_choses(self):
        global gl_DATA
        choses = []
        drops = []
        ck=self.sender()
        #print(self.cks)
        #print('ck',ck)
        number = self.cks.index(ck)
        if ck.checkState() == 2:
            choses.append(number)
        else:
            drops.append(number)
        gl_DATA.set_chose_drop(choses, drops)


class FileDialog(QWidget):
    def __init__(self):
        super().__init__()  # use qwidget to creat window
        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('File dialog')
        # self.show()

    def fontSelect(self):
        # 弹出一个字体选择对话框。getFont()方法
        # 返回一个字体名称和状态信息。状态信息有OK和其他两种
        font, ok = QFontDialog.getFont()

        if ok:
            self.lbl.setFont(font)


class DropDownList(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.lbl = QLabel("Ubuntu", self)

        combo = QComboBox(self)
        combo.addItem("Ubuntu")
        combo.addItem("Mandriva")
        combo.addItem("Fedora")
        combo.addItem("Arch")
        combo.addItem("Gentoo")

        combo.move(50, 50)
        self.lbl.move(50, 150)

        combo.activated[str].connect(self.onActivated)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QComboBox')

    def onActivated(self, text):
        self.lbl.setText(text)
        self.lbl.adjustSize()


if __name__ == '__main__':
    # plt.ion()
    app = QApplication(sys.argv)
    window = Application()
    window.show()  # IMPORTANT!!!!! Windows are hidden by default.
    # 最后，我们进入了应用的主循环中，事件处理器这个时候开始工作。
    # 主循环从窗口上接收事件，并把事件传入到派发到应用控件里。
    # 当调用exit()方法或直接销毁主控件时，主循环就会结束。
    # sys.exit()方法能确保主循环安全退出。外部环境能通知主控件怎么结束。
    # Start the event loop.
    sys.exit(app.exec_())  # exec_()之所以有个下划线，是因为exec是一个Python的关键字。
    # plt.pause(0)
