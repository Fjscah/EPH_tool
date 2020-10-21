#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/10/19 15:22
# @Author : Fjscah
# @Version：V 0.1
# @File : calcium_base.py.py
# @desc :
import  re
import matplotlib.pyplot as plt
import numpy as np
import PIL
import cv2
from tifffile import imread
import  os
import  traceback
from PIL import  Image
from TiffPseudoCapture import VideoCapture
# read the video
import image_base
import  func_base
import threading

class CalciumEvent():
    def __init__(self,filename,freq=20,chose='all'):
        # chose  'all',or list
        self.pathway=filename
        self.sample_freq=freq #(Hz)
        self.chose=chose
        self.assignfile()

    def assignfile(self):
        # find all file and set index
        self.filedicts={}
        f=open(self.pathway, 'r', newline='\n')
        dirname=os.path.dirname(self.pathway)
        line=f.readline()
        while(line):
            #print(line)
            m=re.match('s(\d*).*',line)
            if m:
                #print(m)
                num=int(m.group(1))
                #print(num)
                #----  absolute filename  ---#
                print(line)
                line=func_base.seek_files(dirname,line)
                print('dir',dirname,line)
                if num in self.filedicts:
                    self.filedicts[num].append(line)
                elif self.chose=='all':
                    self.filedicts[num]=[line]
                elif num in self.chose:
                    self.filedicts[num]=[line]
            line=f.readline()
    def load_all_frame(self,event=None):
        self.cur_sweep=0
        self.cur_time=0
        self.contours = [] ## roi
        self.calevent={}
        outfolder='result'
        dirname = os.path.dirname(self.pathway)
        if self.chose!='all':
            for (sweep,starttime) in zip(self.chose,self.starttime):
                if sweep not in self.filedicts:
                    print('dont find sweep frame file !!!!',sweep)
                    continue
                self.cur_sweep=sweep
                self.cur_time=starttime
                for filename in self.filedicts[sweep]:
                    framenumbert=self.load_frame(filename,event)
                    self.cur_time=self.cur_time+framenumber/self.sample_freq
        else:
            self.sweeps=list(self.filedicts.keys())
            self.sweeps.sort()
            for sweep in self.sweeps:
                self.cur_frame=0
                self.cur_sweep=sweep
                for filename in self.filedicts[sweep]:
                    framenumber=self.load_frame(filename)
                    self.cur_frame+=framenumber
                print('sweep : ',sweep,'framenumber',self.cur_frame,'files:',self.filedicts[sweep])
                f=open(os.path.join(dirname,outfolder,'s_'+str(sweep)),'w',newline='\r\n')
                for cal in self.calevent[sweep]:
                    strr='\t'.join(cal)
                    f.write(strr+'\r\n')
                f.close()



    def load_frame(self,filename,event=None):
        # filename='vtest.avi'
        print(filename)
        file, postfix = os.path.splitext(filename)
        print('type', postfix)
        #----  result: show roi result include nummber  ---#
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 500, 500)
        #----  input: show input + soma mask  ---#
        cv2.namedWindow("input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('input', 500, 500)
        # create the subtractor
        his = 40
        fgbg = image_base.CalciumRoi(his=his)
        counter = 0
        #  not avi
        if postfix == '.avi':
            cap = cv2.VideoCapture(filename)
            fps = cap.get(cv2.CAP_PROP_FPS)  # CV_CAP_PROP_FPS
            print('fps', fps)
            while True:
                ret, frame = cap.read()
                print(ret, frame)
                cv2.imwrite("input.png", frame)
                cv2.imshow('input', frame)
                result, m_, c = getPerson(frame)
                counter += c
                cv2.imshow('result', result)
                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    cv2.imwrite("result.png", result)
                    cv2.imwrite("mask.png", m_)
                    break
            cap.release()

        elif postfix == '.tif':
            print("The selected stack is a .tif")
            # ----  print pic info  ---#
            dataset = VideoCapture(filename)
            dataset.show_info()
            # dataset=Image.open(filename)
            w = dataset.get(3)
            h = dataset.get(4)
            n_frame = dataset.get(7) # frame number
            # ----  get first frame , find soma  ---#
            image = imread(filename, key=0)
            print(image.dtype)
            # ----  get soma contours  ---#
            # get first file

            somacontours = image_base.get_soma_contours(image, number=1)  # if negtive put all possible contours
            # th,mask=cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # soma_color=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image, somacontours, -1, np.iinfo(image.dtype).max, -1)  # fill
            # ----  soma info and show soma  ---#
            cv2.namedWindow("soma", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("soma", 500, 500)
            # cv2.putText(soma_color,"Enter N to reset soma region",(50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
            cv2.imshow('soma', image)
            print('len soma', len(somacontours))
            key = cv2.waitKey(0)
            cv2.destroyWindow('soma')
            # ----  soma mask  ---#
            #somamask = np.zeros_like(image)
            somamask = cv2.drawContours(image, somacontours, -1, np.iinfo(image.dtype).max, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
            somamask = cv2.dilate(somamask, kernel)
            # ----  if soma region wrong , reset  ---#
            # todo: wait code
            print('key', key)
            # ----  get every frame  ---#

            self.results = np.zeros((h, w), dtype=np.uint8)
            masks = np.zeros((h, w), dtype=np.uint8)
            for i in range(n_frame):
                frame = imread(filename, key=i)
                # input frame add soma mask
                frame = cv2.drawContours(frame, somacontours, -1, 0, -1)
                # try:
                #     # frame=cv2.normalize(frame,0,np.iinfo(frame.dtype),cv2.NORM_MINMAX)
                cv2.imshow('input', frame * 30)  # too dark
                # except Exception as e:
                #     # 访问异常的错误编号和详细信息
                #     print(e.args)
                #     print(str(e))
                #     print(repr(e))
                #     print(frame.shape)
                #     print(type(frame))
                #     print(traceback.print_exc())
                #  The decay time of GCaMP6f is about 200 ms rise 100ms
                if i < his:
                    fgbg.get_calcium_roi(frame, omit=True)
                else:
                    #
                    m_, c, contours = fgbg.get_calcium_roi(frame, omit=False)
                    if c > 0:
                        print('frame:', i, '   counter:', c)
                        # print(results.shape,result.shape,results.dtype,result.dtype)
                        # results=cv2.bitwise_or(results,result)
                        masks = cv2.bitwise_or(masks, m_)
                        number=self.union_contours(contours)
                        print('frame:', i, '   counter:', number)
                        self.results = np.zeros((h, w), dtype=np.uint8)
                        self.results=cv2.drawContours(self.results,self.contours)
                        for i,cnt in enumerate(self.contours):
                            cv2.putText(self.results, str(i), (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                        (0, 255, 0), 1)
                        if self.cur_sweep not in self.calevent:
                            self.calevent[self.cur_sweep]=[self.cur_frame+i+1]+number
                        else:
                            self.calevent[self.cur_sweep].append([self.cur_frame+i+1]+number)
                    elif i % 100 == 0:
                        print('frame:', i)

                # cv2.imshow('result', results)
                cv2.imshow('result', self.results)
                k = cv2.waitKey(100) & 0xff
                if k == 27:
                    # cv2.imwrite("result.png", result)
                    # cv2.imwrite("mask.png", m_)
                    break
        else:
            print('NONE')

        contours, hierarchy = cv2.findContours(masks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        masks = cv2.drawContours(masks, contours, -1, 255, 3)
        cv2.imshow('result', masks)

        print('total counter', len(contours))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return  n_frame

    def union_contours(self,contours):
        # self.contours  is list
        #----  primary judge two polygon has intersection : compar the first coord  ---#
        print(func_base.interval_length(self.contours))
        print(func_base.interval_length(contours))
        oris=[(cnt[0][0],cnt[0][1]) for cnt in self.contours]
        news=[(cnt[0][0],cnt[0][1]) for cnt in contours]
        absdis=func_base.abs_distance(oris,news)
        candis=np.where(absdis<10)
        numbers=[]
        # union old contours
        for x,y in candis:
            oricontour=self.contours[y]
            newcontour=contours[x]
            newcontour=func_base.union_contours(oricontour,newcontour)
            numbers.append(y)
            if newcontour:
                self.contours[y]=newcontour
            else:
                candis[x,y]=False
        lenn=len(self.contours)
        i=0
        for x,row in enumerate(candis):
            if row==False.all():
                numbers.append(lenn+i)
                self.contours.append(contours[x])
                i+=1
        return numbers



class Calcium():
    def __init__(self,filename,chose,starttime):
        # chose  'all',or list
        self.pathway=filename
        self.chose=chose
        self.assignfile()
        self.starttime=starttime# (second)
        self.sample_freq=20 #(Hz)
    def assignfile(self):
        self.filedicts={}
        f=open(self.pathway, 'r', newline='\n')
        dirname=os.path.dirname(self.pathway)
        line=f.readline()
        while(line):
            #print(line)
            m=re.match('s(\d*).*',line)
            if m:
                #print(m)
                num=int(m.group(1))
                #print(num)
                #----  absolute filename  ---#
                line=func_base.seek_files(dirname,line)
                if num in self.filedicts:
                    self.filedicts[num].append(line)
                elif self.chose=='all':
                    self.filedicts[num]=[line]
                elif num in self.chose:
                    self.filedicts[num]=[line]
            line=f.readline()
    def load_all_frame(self,event):
        self.cur_sweep=0
        self.cur_time=0
        self.contours = []
        if self.chose!='all':
            for (sweep,starttime) in zip(self.chose,self.starttime):
                if sweep not in self.filedicts:
                    print('dont find sweep frame file !!!!',sweep)
                    continue
                self.cur_sweep=sweep
                self.cur_time=starttime
                for filename in self.filedicts[sweep]:
                    framenumber=self.load_frame(filename,event)
                    self.cur_time=self.cur_time+framenumber/self.sample_freq


    def load_frame(self,filename,event):
        # filename='vtest.avi'
        file, postfix = os.path.splitext(filename)
        print('type', postfix)
        #----  result: show roi result include nummber  ---#
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 500, 500)
        #----  input: show input + soma mask  ---#
        # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('input', 500, 500)
        # create the subtractor
        his = 40
        fgbg = image_base.CalciumRoi(his=his)
        counter = 0
        #  not avi
        if postfix == '.avi':
            cap = cv2.VideoCapture(filename)
            fps = cap.get(cv2.CAP_PROP_FPS)  # CV_CAP_PROP_FPS
            print('fps', fps)
            while True:
                ret, frame = cap.read()
                print(ret, frame)
                cv2.imwrite("input.png", frame)
                cv2.imshow('input', frame)
                result, m_, c = getPerson(frame)
                counter += c
                cv2.imshow('result', result)
                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    cv2.imwrite("result.png", result)
                    cv2.imwrite("mask.png", m_)
                    break
            cap.release()

        elif postfix == '.tif':
            print("The selected stack is a .tif")
            # ----  print pic info  ---#
            dataset = VideoCapture(filename)
            dataset.show_info()
            # dataset=Image.open(filename)
            w = dataset.get(3)
            h = dataset.get(4)
            n_frame = dataset.get(7) # frame number
            # ----  get first frame , find soma  ---#
            image = imread(filename, key=0)
            print(image.dtype)
            # ----  get soma contours  ---#
            # get first file

            somacontours = image_base.get_soma_contours(image, number=1)  # if negtive put all possible contours
            # th,mask=cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # soma_color=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image, somacontours, -1, np.iinfo(image.dtype).max, -1)  # fill
            # ----  soma info and show soma  ---#
            cv2.namedWindow("soma", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("soma", 500, 500)
            # cv2.putText(soma_color,"Enter N to reset soma region",(50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
            cv2.imshow('soma', image)
            print('len soma', len(somacontours))
            key = cv2.waitKey(0)
            cv2.destroyWindow('soma')
            # ----  soma mask  ---#
            somamask = np.zeros_like(image)
            somamask = cv2.drawContours(image, somacontours, -1, np.iinfo(image.dtype).max, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
            somamask = cv2.dilate(somamask, kernel)
            # ----  if soma region wrong , reset  ---#
            # todo: wait code
            print('key', key)
            # ----  get every frame  ---#
              #
            self.results = np.zeros((h, w), dtype=np.uint8)
            masks = np.zeros((h, w), dtype=np.uint8)
            for i in range(n_frame):
                frame = imread(filename, key=i)
                # input frame add soma mask
                frame = cv2.drawContours(frame, somacontours, -1, 0, -1)
                try:
                    # frame=cv2.normalize(frame,0,np.iinfo(frame.dtype),cv2.NORM_MINMAX)
                    cv2.imshow('input', frame * 30)  # too dark
                except Exception as e:
                    # 访问异常的错误编号和详细信息
                    print(e.args)
                    print(str(e))
                    print(repr(e))
                    print(frame.shape)
                    print(type(frame))
                    print(traceback.print_exc())
                #  The decay time of GCaMP6f is about 200 ms rise 100ms
                if i < his:
                    fgbg.get_calcium_roi(frame, omit=True)
                else:
                    #
                    m_, c, contours = fgbg.get_calcium_roi(frame, omit=False)
                    if c > 0:
                        print('frame:', i, '   counter:', c)
                        # print(results.shape,result.shape,results.dtype,result.dtype)
                        # results=cv2.bitwise_or(results,result)
                        masks = cv2.bitwise_or(masks, m_)
                        number=self.union_contours(contours)
                        print('frame:', i, '   counter:', number)
                        self.results = np.zeros((h, w), dtype=np.uint8)
                        self.results=cv2.drawContours(self.results,self.contours)
                        for i,cnt in enumerate(self.contours):
                            cv2.putText(self.results, str(i), (cnt[0][0][0], cnt1[0][0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                        (0, 255, 0), 1)
                        event.start()
                    elif i % 100 == 0:
                        print('frame:', i)

                # cv2.imshow('result', results)
                cv2.imshow('result', self.results)
                k = cv2.waitKey(100) & 0xff
                if k == 27:
                    # cv2.imwrite("result.png", result)
                    # cv2.imwrite("mask.png", m_)
                    break
        else:
            print('NONE')

        contours, hierarchy = cv2.findContours(masks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        masks = cv2.drawContours(masks, contours, -1, 255, 3)
        cv2.imshow('result', masks)

        print('total counter', len(contours))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return  n_frame

    def union_contours(self,contours):
        # self.contours  is list
        #----  primary judge two polygon has intersection : compar the first coord  ---#
        oris=[(cnt[0][0],cnt[0][1]) for cnt in self.contours]
        news=[(cnt[0][0],cnt[0][1]) for cnt in contours]
        absdis=func_base.abs_distance(oris,news)
        candis=np.where(absdis<10)
        numbers=[]
        # union old contours
        for x,y in candis:
            oricontour=self.contours[y]
            newcontour=contours[x]
            newcontour=func_base.union_contours(oricontour,newcontour)
            numbers.append(y)
            if newcontour:
                self.contours[y]=newcontour
            else:
                candis[x,y]=False
        lenn=len(self.contours)
        i=0
        for x,row in enumerate(candis):
            if row==False.all():
                numbers.append(lenn+i)
                self.contours.append(contours[x])
                i+=1
        return numbers



if __name__ == '__main__':
    filename ="I:\\Fang\\20141211-15d\\cs2\\VF1\\frame file.txt"
    cal=Calcium(filename,'all')
