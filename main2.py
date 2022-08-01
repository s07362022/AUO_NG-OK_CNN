# from __future__ import print_function

# import matplotlib
# matplotlib.use('Qt5Agg')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from keras.datasets import mnist
import argparse
import anogan2 as anogan
import dist 
import tensorflow as tf
####GPU

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=5)
parser.add_argument('--label_idx', type=int, default=0)
parser.add_argument('--mode', type=str, default='test', help='train, test')
args = parser.parse_args()

### 0. prepare data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.astype(np.float32) - 127.5) / 127.5
IMAGE_SIZE = 300#300
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w = image.shape
    #print(image.shape)
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

### data 
D = "D:\\harden\\dataset\\H1_water_de\\ok\\"   #####path  ok
E = "D:\\harden\\dataset\\H1_water_de\\ng\\"   #####path  ng
images = []
labels = []
test_image = []
test_label = []
dir_counts = 0
IMAGE_SIZE = 300#300
def d (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        
        #try:
        img1 = cv2.imread(D+i,0)
        #img1 = img1[:,:,:,None]
        #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
        img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
        images.append(img1)
        labels.append(dir_counts)
        #except:
            #print("error")
        vou +=1
        if vou >=1000:
            break
    print("A already read")
    return(images,labels)
d(D,images,labels)
d(E,images=test_image,labels=test_label)
label = np.array(labels)
images = np.array(images)
test_imgae = np.array(test_image)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(images, label,test_size=0.1,random_state=42 )#

X_train = (X_train.astype(np.float32) /255.0) #- 127.5) / 127.5
# X_test = (X_test.astype(np.float32) /255.0) #- 127.5) / 127.5
X_test = (test_imgae.astype(np.float32) /255.0) 
test_label= np.array(test_label)
X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]
X_train_testing = X_train[:len(X_test)]  #######################
X_test_original = X_test.copy()

#X_train = X_train[y_train==1]
#X_test = X_test[y_test==1]
print ('train shape:', X_train.shape)
print ('test shape:', X_test.shape)
### 1. train generator & discriminator
if args.mode == 'train':
    Model_d, Model_g = anogan.train(20, X_train) #32

### 2. test generator
generated_img = anogan.generate(25)
img = anogan.combine_images(generated_img)
img = (img*127.5)+127.5
img = img.astype(np.uint8)
img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

### opencv view
# cv2.namedWindow('generated', 0)
# cv2.resizeWindow('generated', 256, 256)
# cv2.imshow('generated', img)
# cv2.imwrite('result_latent_10/generator.png', img)
# cv2.waitKey()

### plt view
# plt.figure(num=0, figsize=(4, 4))
# plt.title('trained generator')
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# exit()

### 3. other class anomaly detection

def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    # ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 28, 28, 1), iterations=500, d=d)
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 300, 300, 1), iterations=500, d=d)

    # anomaly area, 255 normalization
    # np_residual = test_img.reshape(28,28,1) - similar_img.reshape(28,28,1)
    np_residual = test_img.reshape(300,300,1) - similar_img.reshape(300,300,1) #.reshape(300,300,1)
    np_residual = (np_residual + 2)/2

    np_residual = (255*np_residual).astype(np.uint8)
    # original_x = (test_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    #　similar_x = (similar_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    original_x = (test_img.reshape(300,300,1)*127.5+127.5).astype(np.uint8)#.reshape(300,300,1)
    similar_x = (similar_img.reshape(300,300,1)*127.5+127.5).astype(np.uint8)#.reshape(300,300,1)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_COOL) #COLORMAP_RAINBOW COLORMAP_JET
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)

    return ano_score, original_x, similar_x, show


### compute anomaly score - sample from test set
# test_img = X_test_original[y_test==1][30]

### compute anomaly score - sample from strange image
# test_img = X_test_original[y_test==0][30]

### compute anomaly score - sample from strange image
img_idx = args.img_idx
label_idx = args.label_idx
test_img = X_test_original[img_idx]
# test_img = np.random.uniform(-1,1, (28,28,1))
def polt_re(qurey,pred,diff,count):
    fig=plt.figure()
    plt.subplot(1,3,1)
    plt.title('query image')
    plt.imshow(qurey.reshape(300,300), cmap=plt.cm.gray)# .reshape(300,300)
    plt.subplot(1,3,2)
    plt.title('generated similar image')
    plt.imshow(pred.reshape(300,300), cmap=plt.cm.gray) #.reshape(300,300)
    plt.subplot(1,3,3)
    plt.title('anomaly detection')
    plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
    fig.savefig('./out/output{}.png'.format(count),dpi=fig.dpi)
    #plt.show()
print("len(X_test_original): ",len(X_test_original))
# score, qurey, pred, diff = anomaly_detection(test_img)


###########################################################
scorelist2= []
y_train = y_train[:len(X_test_original)]
for i in range(len(X_train_testing)):
    # start = cv2.getTickCount()
    score, qurey, pred, diff = anomaly_detection(X_train_testing[i])    
    scorelist2.append(score)
    # time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d label, %d : done'%(y_train[i], img_idx), 'score: %.2f '%score)#, '%.2fms'%time)
    count = str(i)
    cv2.imwrite('./fake_img/0729_ori/tarin_gqurey{}.png'.format(count) , qurey)
    cv2.imwrite('./fake_img/0729_ori/tarin_pred{}.png'.format(count), pred)
    cv2.imwrite('./fake_img/0729_ori/tarin_diff{}.png'.format(count), diff)
    polt_re(qurey,pred,diff,count)
print("訓練集分數: ",scorelist2)
try:
    f=open("./fake_img/0729_ori/train_OK_0.txt","w")
    for k in range(len(scorelist2)):
        f.write(scorelist2[k]+"\n")
except:
    pass

import pandas as pd
name_list = scorelist2
height_list = y_train
df_train = pd.DataFrame((zip(name_list, height_list)), columns = ['score', 'labe'])
print(df_train)
df_train.to_csv("./anogan_729_train_ori.csv")#scorelist

########### use train score to set gate ############
try:
    df_train=df_train.drop(['Unnamed: 0'],axis=1)
    deviate=int(len(df_train)*0.05)
    for j in range(deviate):
        df_train=df_train.drop(df_train['score'].idxmax())
    train_value=df_train['score'].values.tolist()
    train_mean=np.mean(train_value)
    train_var=np.var(train_value,ddof=1)
    train_std=np.std(train_value,ddof=1)
    print("train mean=",train_mean,"train variance=",train_var)
    print("OK_mean: ",train_mean)
    print("OK_var: ",train_var)
    print("OK_std: ",train_std)
    cat=int(train_mean+train_std) 
except:
    print("DataFrame Error")
    
    
########### use train score to set gate ############


## matplot view
# plt.figure(1, figsize=(3, 3))
# plt.title('query image')
# plt.imshow(qurey.reshape(300,300), cmap=plt.cm.gray)

# print("anomaly score : ", score)
# plt.figure(2, figsize=(3, 3))
# plt.title('generated similar image')
# plt.imshow(pred.reshape(300,300), cmap=plt.cm.gray)

# plt.figure(3, figsize=(3, 3))
# plt.title('anomaly detection')
# plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
# plt.show()
scorelist= []
testNG_list = []
testall_list=[]
for i in range(len(X_test_original)):
    # start = cv2.getTickCount()
    score, qurey, pred, diff = anomaly_detection(X_test_original[i])    
    scorelist.append(score)
    if cat <= score:
        print("This is NG sample")
        testNG_list.append(1)
        testall_list.append(1)
        # time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('%d label, %s : done'%(test_label[i], "it is NG"), 'score: %.2f '%score)#, '%.2fms'%time)
    else:
        testall_list.append(0)
    count = str(i)
    cv2.imwrite('./fake_img/0729_ori/gqurey{}.png'.format(count) , qurey)
    cv2.imwrite('./fake_img/0729_ori/pred{}.png'.format(count), pred)
    cv2.imwrite('./fake_img/0729_ori/diff{}.png'.format(count), diff)
    polt_re(qurey,pred,diff,count)
    print("NG 數量有 %d 個 , OK 數量有 %d 個"%(len(testNG_list),(len(testall_list)-len(testNG_list))))
    
print("測試NG集分數: ",scorelist)

try:
    f=open("./fake_img/0729_ori/test_NG_0.txt","w")
    for k in range(len(scorelist)):
        f.write(scorelist[k]+"\n")
except:
    pass

name_list = scorelist
height_list = test_label
df = pd.DataFrame((zip(name_list, height_list)), columns = ['score', 'labe'])
print(df)
df.to_csv("./anogan_729_test_ori.csv")
### 4. tsne feature view




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#print(metrics.classification_report(y_test_label, prediction))
try:
    import itertools
    import sklearn.metrics as metrics
    cnf_matrix = metrics.confusion_matrix(test_label,testall_list)#y_test_label
    target_names = ['OK', 'NG']
    # plot_confusion_matrix(predict_y)
    plot_confusion_matrix(cnf_matrix, classes=target_names)
    # Calculate the sensitivity and specificity
    TP = confusion_matrix(test_label,testall_list, labels=[1, 0])[0, 0]
    FP = confusion_matrix(test_label,testall_list, labels=[1, 0])[1, 0]
    FN = confusion_matrix(test_label,testall_list, labels=[1, 0])[0, 1]
    TN = confusion_matrix(test_label,testall_list, labels=[1, 0])[1, 1]
    print("True positive: {}".format(TP))
    print("False positive: {}".format(FP))
    print("False negative: {}".format(FN))
    print("True negative: {}".format(TN))
    ############################
    sensitivity = TP/(FN+TP)
    specificity = TN/(TN+FP)
    ################################
    print("Sensitivity: {}".format(sensitivity))
    print("Specificity: {}".format(specificity))
except:
    pass





### t-SNE embedding 
### generating anomaly image for test (radom noise image)

from sklearn.manifold import TSNE

random_image = np.random.uniform(0, 1, (100, 300, 300, 1)) #(100, 300, 300, 1)
#random_image = np.random.uniform(0, 1, (100, 28, 28, 1))
print("random noise image")
plt.figure(4, figsize=(2, 2))
plt.title('random noise image')
plt.imshow(random_image[0].reshape(300,300), cmap=plt.cm.gray) #.reshape(300,300)
#plt.imshow(random_image[0].reshape(28,28), cmap=plt.cm.gray)
# intermidieate output of discriminator
model = anogan.feature_extractor()
feature_map_of_random = model.predict(random_image, verbose=1)
feature_map_of_minist = model.predict(X_test_original[:300], verbose=1)
feature_map_of_minist_1 = model.predict(X_test[:100], verbose=1)

# t-SNE for visulization
output = np.concatenate((feature_map_of_random, feature_map_of_minist, feature_map_of_minist_1))
output = output.reshape(output.shape[0], -1)
anomaly_flag = np.array([1]*100+ [0]*300)

X_embedded = TSNE(n_components=2).fit_transform(output)
plt.figure(5)
plt.title("t-SNE embedding on the feature representation")
plt.scatter(X_embedded[:100,0], X_embedded[:100,1], label='random noise(anomaly)')
plt.scatter(X_embedded[100:300,0], X_embedded[100:300,1], label='mnist(anomaly)')
plt.scatter(X_embedded[300:,0], X_embedded[300:,1], label='mnist(normal)')
plt.legend()
plt.show()



#dist.main_diff(qurey,pred)###diff 