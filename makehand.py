import cv2 
import os 

#img = cv2.imread("D:\\harden\\dataset\\CF_Marco\\ok\\Basler_acA4096-11gm__40092815__20220401_143311279_0019.tiff")
#img2 = img[370:1900,1200:2500]
D = "D:\\harden\\dataset\\H1_water_de\\NG\\"
F= "D:\\harden\\dataset\\H1_water_de\\ng_half\\"

vou=0
for i in os.listdir(D): 


        
    img1 = cv2.imread(D+i)
    h,w=img1.shape[0],img1.shape[1]
    img2 = img1[:int(h/2),:]
    cv2.imwrite(F+"%s_ngtop.jpg" %str(vou), img2) 
    img3 = img1[int(h/2):,:]
    cv2.imwrite(F+"%s_ngdowm.jpg" %str(vou), img3 ) 
    

    vou +=1
    if vou >=500:
        break
#cv2.imshow("ori",img)
#cv2.imshow("img2",img2)
#img_name="D:\\harden\\dataset\\CF_Marco\\CF_cut_ok\\0.jpg"
#cv2.imwrite(img_name, img2) 
#cv2.waitKey(0)
#cv2.destroyAllWindows()