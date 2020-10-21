#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/10/9 12:00
# @Author : Fjscah
# @Version：V 0.1
# @File : TiffPseudoCapture.py
# @desc : packing tiff load like vedio

from tifffile import imread
from cv2 import VideoCapture as Vcap
from PIL import Image
import  numpy as  np
import cv2

import exifread




class VideoCapture(Vcap):
    def __init__(self,filename,channel=1):
        super(VideoCapture,self).__init__()
        self.filename=filename
        dataset = Image.open(filename)
        self.gets=[None]*15
        h, w = np.shape(dataset)
        self.gets[3]=w #CAP_PROP_FRAME_WIDTH
        self.gets[4]=h #CAP_PROP_FRAME_HEIGHT
        self.gets[7]=dataset.n_frames # CAP_PROP_FRAME_COUNT
        self.gets[8]=dataset.format # TIFF , CAP_PROP_FORMAT
        self.gets[6]=dataset.mode #I;16 CAP_PROP_FOURCC

        self.gets[1]=0 # 0 - endframes-1 CAP_PROP_POS_FRAMES
        self.gets[2]=0 # 0 - 100% CAP_PROP_POS_AVI_RATIO
        self.channel=channel
    def get(self,opt=0):
        return self.gets[opt]
    def read(self):
        image=imread(self.filename,key=self.gets[1])
        image.dtype=np.float32
        if image.any():
            if self.channel==3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return True,image
        else:
            return False,None
    def seek(self,frame):
        self.gets[1]=frame

    def tell(self):
        return  self.gets[1]
    def show_info(self):
        print('width:',self.gets[3])
        print('height:',self.gets[4])
        print('frames number:',self.gets[7])
        print('file type:',self.gets[8])
        print('datatype:',self.gets[6])
        print('current frame:',self.gets[1])
    def show_all_info(self):
        # Open image file for reading (binary mode)
        f = open(self.filename, 'rb')

        # Return Exif tags
        tags = exifread.process_file(f)

        # Print the tag/ value pairs
        for tag in tags.keys():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                print("Key: %s, value %s" % (tag, tags[tag]))


