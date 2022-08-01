
import load_dataset as ld 

D1 = "D:\\harden\\dataset\\H1_water_de\\OK2\\"
E = "D:\\harden\\dataset\\H1_water_de\\water_cut_NG\\"
F =  "D:\\harden\\dataset\\CF_Marco\\CF_cut_ok\\"
B = "D:\\harden\\dataset\\CF_Marco\\CF_cut_ng\\"
#S = "E:\\workspace\\project_\\file_smoke"
#C = "E:\\workspace\\segmentation\\unt01_OK\\cat"
test_ldx=[]
test_ldy=[]
ld.d(D1,test_ldx,test_ldy)
ld.d(E,test_ldx,test_ldy,1)
ld.d(F,test_ldx,test_ldy,2)
ld.d(B,test_ldx,test_ldy,3)
import numpy as np
test_ldx=np.array(test_ldx)
test_ldx=(test_ldx*255).astype(np.uint8)
print(test_ldx,test_ldy[0])


'''
import matplotlib.pyplot as plt
import cv2 
import numpy as np
img1 = cv2.imread("D:\\harden\\dataset\\H1_water_de\\OK2\\37.jpg",0)
img2 = cv2.imread("D:\\harden\\dataset\\H1_water_de\\water_cut_NG\\37.jpg",0)
IMAGE_SIZE = 300
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w , _= image.shape
    
    #adj(w,h)
    longest_edge = max(h, w)    
    
    #size = n*n 
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    BLACK = [0, 0, 0]   
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    return cv2.resize(constant, (height, width))
img1=np.expand_dims(img1.astype(np.uint8),axis=-1)
img2=np.expand_dims(img2.astype(np.uint8),axis=-1)
img1=resize_image(img1,IMAGE_SIZE,IMAGE_SIZE)
img2=resize_image(img2,IMAGE_SIZE,IMAGE_SIZE)

np_residual = img2.reshape(300,300,1) - img1.reshape(300,300,1)
np_residual = (np_residual + 2)/4
residual_color = cv2.applyColorMap(np_residual.astype(np.uint8), cv2.COLORMAP_JET)
img1=img1.reshape(1,300,300)
show = cv2.addWeighted(img1.astype(np.uint8), 0.4, residual_color.astype(np.uint8), 0.6, 0.)
plt.imshow(show)
plt.show()
'''