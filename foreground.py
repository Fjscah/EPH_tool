#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/10/7 20:46
# @Author : Fjscah
# @Version：V 0.1
# @File : foreground.py
# @desc :

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



filename='F:\project\python\learn opencv\s19-LED2-100ms_5_MMStack.ome.tif'
#filename='vtest.avi'
file,postfix=os.path.splitext(filename)
print('type',type)
cv2.namedWindow("result",cv2.WINDOW_NORMAL)
cv2.resizeWindow('result',500,500)
cv2.namedWindow("input",cv2.WINDOW_NORMAL)
cv2.resizeWindow('input',500,500)
# create the subtractor
his=40
fgbg=image_base.CalciumRoi()
counter=0
if postfix=='.avi':
    cap = cv2.VideoCapture(filename)
    fps=cap.get(cv2.CAP_PROP_FPS)#CV_CAP_PROP_FPS
    print('fps',fps)
    while True:
        ret, frame = cap.read()
        print(ret, frame)
        cv2.imwrite("input.png", frame)
        cv2.imshow('input', frame)
        result, m_,c = getPerson(frame)
        counter+=c
        cv2.imshow('result', result)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            cv2.imwrite("result.png", result)
            cv2.imwrite("mask.png", m_)
            break
    cap.release()
elif postfix=='.tif':
    print("The selected stack is a .tif")
    #----  print pic info  ---#
    dataset=VideoCapture(filename)
    dataset.show_info()
    #dataset=Image.open(filename)
    w=dataset.get(3)
    h=dataset.get(4)
    number=dataset.get(7)
    #----  get first frame , find soma  ---#
    image=imread(filename,key=0)
    print(image.dtype)
    #----  get soma contours  ---#
    somacontours=image_base.get_soma_contours(image,number=1) # if negtive put all possible contours
    #th,mask=cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #soma_color=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image, somacontours, -1, np.iinfo(image.dtype).max, -1) #fill
    #----  soma info and show soma  ---#
    cv2.namedWindow("soma",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("soma",500,500)
    #cv2.putText(soma_color,"Enter N to reset soma region",(50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
    cv2.imshow('soma',image)
    print('len soma',len(somacontours))
    key=cv2.waitKey(0)
    cv2.destroyWindow('soma')
    #----  soma mask  ---#
    somamask=np.zeros_like(image)
    somamask=cv2.drawContours(image,somacontours,-1,np.iinfo(image.dtype).max, -1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    somamask=cv2.dilate(somamask,kernel)
    #----  if soma region wrong , reset  ---#
    # todo: wait code
    print('key',key)
    #----  get every frame  ---#
    contours=[]#
    results=np.zeros((h,w),dtype=np.uint8)
    masks=np.zeros((h,w),dtype=np.uint8)
    for i in range(number):
        frame = imread(filename,key=i)
        # input frame add soma mask
        frame=cv2.drawContours(frame,somacontours,-1,0,-1)
        try:
            #frame=cv2.normalize(frame,0,np.iinfo(frame.dtype),cv2.NORM_MINMAX)
            cv2.imshow('input',frame*30) # too dark
        except Exception as e:
            # 访问异常的错误编号和详细信息
            print(e.args)
            print(str(e))
            print(repr(e))
            print(frame.shape)
            print(type(frame))
            print(traceback.print_exc())
        #  The decay time of GCaMP6f is about 200 ms rise 100ms
        if i<his:
            fgbg.get_calcium_roi(frame,omit=True)
        else:
            result, m_,c,contour = fgbg.get_calcium_roi(frame,omit=False)
            if c>0:
                print('frame:',i, '   counter:',c)
                # print(results.shape,result.shape,results.dtype,result.dtype)
                # results=cv2.bitwise_or(results,result)
                masks=cv2.bitwise_or(masks,m_)
        if i%100==0:
            print('frame:',i)
        #cv2.imshow('result', results)
        cv2.imshow('result',masks)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            # cv2.imwrite("result.png", result)
            # cv2.imwrite("mask.png", m_)
            break
else:
    print('NONE')


contours, hierarchy = cv2.findContours(masks, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
masks=cv2.drawContours(masks,contours,-1,255,3)
cv2.imshow('result',masks)


print('total counter',len(contours))
cv2.waitKey(0)
cv2.destroyAllWindows()
