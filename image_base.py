import matplotlib.pyplot as plt
import numpy as np
import cv2

import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import PIL
import cv2
from tifffile import imread


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    ImageTk.PhotoImage(Image.fromarray(image))
    return image


class ImageTk_fig():
    def __init__(self,data,color='blue'):
        plt.figure(figsize=(0.2,0.2),facecolor="#FFDAB9")
        ax=plt.subplot(111,)
        ax.spines['top'].set_visible(False) #去掉上边框
        ax.spines['bottom'].set_visible(False) #去掉下边框
        ax.spines['left'].set_visible(False) #去掉左边框
        ax.spines['right'].set_visible(False) #去掉右边框
        plt.plot(data,linewidth=0.5,color=color)

        fig = plt.gcf()

        #plt.imshow(***[等待保存的图片]***)
        #fig.pixels(100,100)
        #fig.set_size_inches(19.2/3,10.8/3) #dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.gca().patch.set_alpha(0.5)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        fig.savefig('target.png', format='png', transparent=True, dpi=300, pad_inches = 0)
        plt.clf()
        self.image= ImageTk.PhotoImage(file ='target.png' )


class App(tk.Tk):
    def __init__(self,fig,frame):
        super().__init__()

        self.lbPic = tk.Label(self, text='test', width=400, height=600)


        self.im_orig = cv2.imread('promo_02.png')

        self.xmin_orig = 8
        self.ymin_orig = 12
        self.xmax_orig = 352
        self.ymax_orig = 498

        # cv2.rectangle(
        #     self.im_orig,
        #     pt1 = (self.xmin_orig, self.ymin_orig),
        #     pt2 = (self.xmax_orig, self.ymax_orig),
        #     color = (0, 255, 0),
        #     thickness = 2
        # )

        self.im_orig = self.im_orig[:, :, ::-1]  # bgr => rgb   necessary

        tkim = ImageTk.PhotoImage(Image.fromarray(self.im_orig))
        self.lbPic['image'] = tkim
        self.lbPic.image = tkim

        self.lbPic.bind('<Configure>', self.changeSize)
        self.lbPic.pack(fill=tk.BOTH, expand=tk.YES)

    def changeSize(self, event):
        im = cv2.resize(self.im_orig, (event.width, event.height))

        tkim = ImageTk.PhotoImage(Image.fromarray(im))
        self.lbPic['image'] = tkim
        self.lbPic.image = tkim

def get_soma_contours(image, number= 1) :
    """

    @type image: np.array single channel
    @type number:int
    """
    # transform data to uint 8
    if image.dtype==np.uint8:
        pass
    elif image.dtype==np.uint16:
        image = image / 256
        image = np.array(image, dtype=np.uint8)
    elif image.dtype==np.uint64:
        image=np.imag/(2**56)
        image=np.array(image,dtype=np.uint8)
    
    # th,mask=cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #----  erosion axon and dendrites  ---#
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20), (-1, -1))
    mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, rect)
    #----  dilate soma  ---#
    kernel = np.ones((60, 60), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    #----  get soma candidate , set 255   ---#
    th, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #----  get soma candidate contours  ---#
    somacontours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if number<0:
        return somacontours
    else:
        areas=[]
        somacontour = []
        # ----  sort according area  ---#
        for soma in somacontours:
            area = cv2.contourArea(soma)
            areas.append((area,soma))
            areas.sort(key=lambda x:x[0])
            areas=areas[:number]
        return [x[1] for x in areas]


class CalciumRoi(object):
    def __init__(self,his=40,buffermasklen=3):
        # create the subtractor
        self.his = his
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.his, varThreshold=50, detectShadows=False)
        self.lastmask=None
        self.buffermasklen=buffermasklen
        self.buffermask=[]
        self.maskname='mask'
        #self.resultmask='mask number'
        cv2.namedWindow(self.maskname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.maskname, 500, 500)
        #cv2.namedWindow(self.resultmask,cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(self.resultmask,500,500)

    def get_calcium_roi(self,frame,omit=False):

        # get the front mask
        mask = self.fgbg.apply(frame)
        # skip this frame
        if omit:
            return  0
        #----  eliminate noise  ---#
        # if duration < 2 frame and region < 5 pixel
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3),(-1,-1)) # make sure kernel odd
        # duration > 2 frame
        nowmask2=cv2.erode(mask,kernel)
        if self.buffermask:
            nowmask1=mask.copy()
            for buffermask in self.buffermask:
                dilationlask=cv2.dilate(buffermask,kernel,iterations=1)
                nowmask1=cv2.bitwise_and(dilationlask,nowmask1)
                # region > 3 pixel
            nowmask=cv2.bitwise_or(nowmask1,nowmask2)
        else:
            nowmask=nowmask2

        #----  update last frame  ---#
        self.buffermask.append(mask)
        if len(self.buffermask)>self.buffermasklen:
            self.buffermask.pop(0)


        #----  show contour and mask  ---#

        cv2.imshow(self.maskname, nowmask)
        # find the max area contours
        contours, hierarchy = cv2.findContours(nowmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        c=len(contours)
        if c>100:
             print('too much')
        #     return 0,0,0,0
        # result=np.zeros_like(frame,np.uint8)
        # result2text=result.copy()
        # #print('-----',result.shape,image.shape)
        # # print('candidate contours',len(contours))
        # c=0
        # for c,cnt in enumerate((contours),1):
        #     result=cv2.drawContours(result,contours,-1,255,-1)
            #result2text=cv2.drawContours(result,contours,-1,255,5)
            #           照片   /添加的文字    /左下角坐标  /字体                            /字体大小 /颜色            /字体粗细
            #cv2.putText(result2text, str(c), (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 5)
            # rect = cv2.minAreaRect(contours[c])
            # cv2.ellipse(image, rect, (0, 255, 0), 2, 8)
            #cv2.circle(result, (np.int32(rect[0][0]), np.int32(rect[0][1])), 20, (255, 0, 0), 2, 8, 0)
        #cv2.imshow(self.resultmask,result2text)
        # if c>0:
        #      print('frame counter',c)
        # else:
        #     print(counter)
        # print('valid contours:',counter)
        return nowmask,c,contours

def main():
    app = App()
    app.title('缩放图像')

    app.mainloop()


if __name__ == '__main__':
    main()