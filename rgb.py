import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('D:\\harden\\dataset\\H1_water_de\\resize_ori\\reok_1.jpg')
img1=img1[:,146:156]
img2 = cv2.imread('D:\\harden\\dataset\\H1_water_de\\resize_ori\\oriok_1.jpg')
img3 = cv2.imread('D:\\harden\\dataset\\H1_water_de\\resize_ori\\oriok_1.jpg')
img3= cv2.resize(img3,(300,300))
#img3 = cv2.imread('D:\\harden\\dataset\\H1_water_de\\ok_half\\0_dowm.jpg')
#img4 = cv2.imread('D:\\harden\\dataset\\H1_water_de\\ok_half\\0_top.jpg')
#img5 = cv2.imread('D:\\harden\\siameNet\\0727\\pair0_1.png') #D:\harden\dataset\H1_water_de\OK2
#img5=img5[:,133:168]
#img6 = cv2.imread('D:\\harden\\dataset\\H1_water_de\\half_resize\\reok_0.jpg')
#img6=img6[:,139:160]
#img7 = cv2.imread('D:\\harden\\dataset\\H1_water_de\\half_resize\\reok_1.jpg')
#img7=img7[:,139:160]
# 畫出 RGB 三種顏色的分佈圖
color = ('b','g','r')
print(img1.shape)

for i, col in enumerate(color):
    #histr1 = cv2.calcHist([img1],[i],None,[256],[0, 256])
    #histr2 = cv2.calcHist([img2],[i],None,[256],[0, 256])
    histr1 = cv2.calcHist([img1],[i],None,[256],[0, 256])
    histr2 = cv2.calcHist([img2],[i],None,[256],[0, 256])
    histr3 = cv2.calcHist([img3],[i],None,[256],[0, 256])
    #histr4 = cv2.calcHist([img4],[i],None,[256],[0, 256])
    #histr6 = cv2.calcHist([img6],[i],None,[256],[0, 256])
    #histr7 = cv2.calcHist([img7],[i],None,[256],[0, 256])
    #plt.subplot(121)
    #plt.imshow(img1[:,128:178])
    #plt.title("ok")
    #plt.subplot(122)
    #plt.imshow(img2[:,128:178])
    #plt.title("ng")
    
    plt.subplot(111)
    plt.plot(histr1, color = col[0],marker="*",label="$mineresize_ok$")
    plt.xlim([0,256])
    plt.title("histr_ok")
    #plt.subplot(224)
    
    plt.plot(histr2,marker="*",label="$ori_ok$")
    plt.title("histr_ok")
    plt.xlim([0,256])
    
    plt.plot(histr3,marker="*",label="$ori_cv2resize$")
    plt.title("histr_ok")
    plt.xlim([0,256])
    
    #plt.plot(histr4,marker="*",label="$ori_okhalf_top$")
    #plt.title("histr_ok")
    #plt.xlim([0,256])
    
    #plt.plot(histr6,marker="*",label="$re_okhalf_down$")
    #plt.title("histr_ok")
    #plt.xlim([0,256])
    
    #plt.plot(histr7,marker="*",label="$re_okhalf_top$")
    #plt.title("histr_ok")
    #plt.xlim([0,256])
    
    plt.legend()
plt.show()
