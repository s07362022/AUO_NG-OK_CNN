#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system(' pip install tensorflow==2.5.0')


# In[2]:


import os
import tensorflow.keras as keras
from tensorflow.keras import layers
import argparse
import tensorflow.keras.backend as K
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras as keras
import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
# In[3]:
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

width = 224
height = 224
channels = 1

niter = 12000
bz = 16
# In[4]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

# Encoder
x = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='conv_1', kernel_regularizer = 'l2')(input_layer)
x = layers.LeakyReLU(name='leaky_1')(x)

x = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(name='leaky_2')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_3', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(name='leaky_3')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_4', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(name='leaky_4')(x)

x = layers.GlobalAveragePooling2D(name='g_encoder_output')(x)

g_e = tensorflow.keras.models.Model(inputs=input_layer, outputs=x)

g_e.summary()


# In[5]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

x = g_e(input_layer)

y = layers.Dense(width * width * 2, name='dense')(x) # 2 = 128 / 8 / 8
y = layers.Reshape((width//8, width//8, 128), name='de_reshape')(y)

y = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', name='deconv_1', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_1')(y)

y = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='deconv_2', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_2')(y)

y = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='deconv_3', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_3')(y)

y = layers.Conv2DTranspose(channels, (1, 1), strides=(1,1), padding='same', name='decoder_deconv_output', kernel_regularizer = 'l2', activation='tanh')(y)

g = tensorflow.keras.models.Model(inputs=input_layer, outputs=y)

g.summary()


# In[6]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

z = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='encoder_conv_1', kernel_regularizer = 'l2')(input_layer)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='encoder_conv_2', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_1')(z)
z = layers.LeakyReLU()(z)


z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='encoder_conv_3', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_2')(z)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_41', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_3')(z)
z = layers.LeakyReLU()(z)

z = layers.GlobalAveragePooling2D(name='encoder_output')(z)

encoder = tensorflow.keras.models.Model(input_layer, z)
encoder.summary()


# In[7]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

f = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='f_conv_1', kernel_regularizer = 'l2')(input_layer)
f = layers.LeakyReLU(name='f_leaky_1')(f)

f = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='f_conv_2', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_1')(f)
f = layers.LeakyReLU(name='f_leaky_2')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_3', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_2')(f)
f = layers.LeakyReLU(name='f_leaky_3')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_4', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_3')(f)
f = layers.LeakyReLU(name='feature_output')(f)

feature_extractor = tensorflow.keras.models.Model(input_layer, f)

feature_extractor.summary()


# In[8]:


class AdvLoss(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class CntLoss(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class EncLoss(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(g_e(ori) - encoder(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
# model for training
input_layer = layers.Input(name='input', shape=(height, width, channels))
gan = g(input_layer) # g(x)

adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cnt_loss = CntLoss(name='cnt_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])

gan_trainer = tensorflow.keras.models.Model(input_layer, [adv_loss, cnt_loss, enc_loss])

# loss function
def loss(yt, yp):
    return yp

losses = {
    'adv_loss': loss,
    'cnt_loss': loss,
    'enc_loss': loss,
}

lossWeights = {'cnt_loss': 20.0, 'adv_loss': 1.0, 'enc_loss': 1.0}

# compile
op= Adam(lr=0.0002)
gan_trainer.compile(optimizer = op, loss=losses, loss_weights=lossWeights)


# In[9]:


input_layer = layers.Input(name='input', shape=(height, width, channels))

f = feature_extractor(input_layer)

d = layers.GlobalAveragePooling2D(name='glb_avg')(f)
d = layers.Dense(1, activation='sigmoid', name='d_out')(d)
    
d = tensorflow.keras.models.Model(input_layer, d)
d.summary()


# In[10]:


op= Adam(lr=0.00008)
d.compile(optimizer=op, loss='binary_crossentropy')


# In[11]:


IMAGE_SIZE=224
D = "D:\\harden\\dataset\\H1_water_de\\OK2\\"
E = "D:\\harden\\dataset\\H1_water_de\\water_cut_NG\\"
#test_final = "E:\\workspace\\opencv_class\\final_test\\test\\"
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w = image.shape
    
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


images = []
fail=[]
labels = []
fail_label=[]
dir_counts = 0
def d1 (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        #print("I:",D+i)
        #cv2.imread(D+i,0)
        try:
            #print("P")
            img1 = cv2.imread(D+i,0)
            #print("P")
            #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            #print("P")
            img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            labels.append(dir_counts)
        except:
            print("error")
        vou +=1
        if vou >=1000:
            break
    print("A already read")
    return(images,labels)

def d2 (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        #print("I:",D+i)
        #cv2.imread(D+i,0)
        try:
            #print("P")
            img1 = cv2.imread(D+i,0)
            #print("P")
            # img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            #print("P")
            img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            labels.append(dir_counts+1)
        except:
            print("error")
        vou +=1
        if vou >=1000:
            break
    print("A already read")
    return(images,labels)


# In[12]:
def d3 (D=D,images=images,labels=labels):
    vou=0
    for i in os.listdir(D):
        #print("I:",D+i)
        #cv2.imread(D+i,0)
        try:
            #print("P")
            img1 = cv2.imread(D+i,0)
            #print("P")
            #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            #print("P")
            img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            labels.append(dir_counts)
        except:
            print("error")
        vou +=1
        if vou >=100:
            break
    print("A already read")
    return(images,labels)

test_1,test_2=d2(E,images=fail,labels=fail_label)
d3(D,images=fail,labels=fail_label)
train_x1,trainx2=d1(D)
train_x1,trainx2=np.expand_dims(train_x1,axis=-1),np.expand_dims(trainx2,axis=-1) 


# In[13]:


print("len(train_x1)",len(train_x1))


# In[14]:


test_1,test_2=np.expand_dims(test_1,axis=-1),np.expand_dims(test_2,axis=-1)


# In[15]:

print("len(test_1)",len(test_1))


# In[16]:


from sklearn.model_selection import train_test_split
label = np.array(labels)
X_train_img,X_test_img,y_train_label,y_test_label =  train_test_split(train_x1, trainx2,test_size=0.1,random_state=42 )#
X_train = np.array(X_train_img, dtype=np.float32)
X_test = np.array(X_test_img, dtype=np.float32)
#print("X_train.shape",X_train.shape)
x_train_std = X_train  /255.0#/255.0     / 127 - 1
x_test_std  =  X_test  /255.0#/255.0 / 127 - 1
x_ok = x_train_std
x_test = x_test_std 
#y_trainOneHot = np_utils.to_categorical(y_train_label)
#y_testOneHot = np_utils.to_categorical(y_test_label)
print("x_train_std.shape",x_train_std.shape)
#print("y_train_label",y_train_label.shape)


# In[17]:


#from keras.datasets import mnist
import cv2
import numpy as np

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_ok = x_train[y_train == 1] # 6742 筆
# x_test = x_test[(y_test == 7) | (y_test == 1)] # 1135 筆 "1", 1028 筆 "7"
# y_test = y_test[(y_test == 7) | (y_test == 1)]

# def reshape_x(x):
#     new_x = np.empty((len(x), width, height))
#     for i, e in enumerate(x):
#         new_x[i] = cv2.resize(e, (width, height))
#     return np.expand_dims(new_x, axis=-1) / 127 - 1
  
# x_ok = reshape_x(x_ok)
# x_test = reshape_x(x_test)


# In[18]:


x_ok.max()


# In[19]:





# In[20]:


def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        np.random.shuffle(idxes)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y]


# In[21]:


train_data_generator = get_data_generator(x_ok, bz)


# In[ ]:





# In[ ]:

if args.mode == 'train':
    for i in range(niter):
        
        ### get batch x, y ###
        x, y = train_data_generator.__next__()
            
        ### train disciminator ###
        d.trainable = True
            
        fake_x = g.predict(x)
            
        d_x = np.concatenate([x, fake_x], axis=0)
        d_y = np.concatenate([np.zeros(len(x)), np.ones(len(fake_x))], axis=0)
            
        d_loss = d.train_on_batch(d_x, d_y)

        ### train generator ###
        
        d.trainable = False        
        g_loss = gan_trainer.train_on_batch(x, y)
        
        if i % 500 == 0:
            print(f'niter: {i+1}, g_loss: {g_loss}, d_loss: {d_loss}')
            # save weights for each epoch
            g.save_weights('weights/Anogenerator.h5', True)
            d.save_weights('weights/Anodiscriminator.h5', True)#g_e
            g_e.save_weights('weights/Anog_e.h5', True)#feature_extractor
            feature_extractor.save_weights('weights/feature_extractor.h5', True)
            print("model save ")
'''
def load_model():
    d.load_weights('./weights/Anodiscriminator.h5')
    g.load_weights('./weights/Anogenerator.h5')
    g_e.load_weights('./weights/Anog_e.h5')
    feature_extractor.load_weights('weights/feature_extractor.h5')
    return g, d,g_e,feature_extractor
'''    
if args.mode=='test':
    d.load_weights('./weights/Anodiscriminator.h5')
    g.load_weights('./weights/Anogenerator.h5')
    g_e.load_weights('./weights/Anog_e.h5')
    feature_extractor.load_weights('weights/feature_extractor.h5')

# In[ ]:
test_1=test_1/255.0
encoded = g_e.predict(test_1)
encoded_gan = g_e.predict(g.predict(test_1))
gan_x = g.predict(test_1)
score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~

'''

# encoded = g_e.predict(x_test)
encoded = g_e.predict(test_1)
# gan_x = g.predict(x_test)
gan_x = g.predict(test_1)
encoded_gan = g_e.predict(gan_x)
score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1
'''
#try:
for k in range(len(gan_x)):
    img_gan=(gan_x[k]*255.0).astype(np.uint8)#*255.0
    cv2.imwrite("D:\\harden\\gan\\GAN-main\\Ganomaly_fake\\727\\272_{}.png".format(k),img_gan)
#except:
    #pass


# In[ ]:

print("gan_x",gan_x[0]*255.0)

f=open("D:\\harden\\gan\\GAN-main\\Ganomaly_re\\test_3.txt","w")
for k in range(len(score)):
    f.write("score=  "+str(score[k])+"\n")
    f.write("test_label=  "+str(test_2[k])+"\n")



# In[ ]:


print(score)


# In[ ]:


back=[]
wwww=[]
all_intscore = []
for i in score:
    if i <0.5:
        back.append(0)
        all_intscore.append(0)
    else:
        wwww.append(1)
        all_intscore.append(1)


# In[ ]:


from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 5
print("len()test_1",len(test_1),"len(score)",len(score))
plt.scatter(range(len(test_1)), score, c=['skyblue' if x == 1 else 'pink' for x in test_2]) #y_test])
plt.show()

# In[ ]:


def plot_image(image,labels,prediction,idx,num=10):  

    fig = plt.gcf() 

    fig.set_size_inches(18, 24) 

    if num>25: 

        num=25 

    for i in range(0, num): 

        ax = plt.subplot(5,5, 1+i) 

        ax.imshow(image[idx], cmap='binary') 

        title = "label=" +str(labels[idx]) 

        if len(prediction)>0: 

            title+=",perdict="+str(prediction[idx]) 

        ax.set_title(title,fontsize=10) 

        ax.set_xticks([]);ax.set_yticks([]) 

        idx+=1 

    plt.show() 


# In[ ]:




# In[ ]:


gan_x2=(gan_x*255.0).astype(np.uint8)
plot_image(gan_x2,test_2,score,0,len(test_2))


# In[ ]:


test_1=(test_1*255).astype(np.uint8)
plot_image(test_1,test_2,score,0,len(test_2))


# In[ ]:


i = 1 # or 1
image2 = (gan_x[0]*255.0).astype(np.uint8)
plt.title("gan_x")
plt.imshow(image2.astype(np.uint8), cmap='gray')
plt.show()

# In[ ]:


image = (test_1[0]*255.0).astype(np.uint8)#x_test 64,64
image = image * 255.0 #+ 127
plt.title("ori")
plt.imshow(image.astype(np.uint8), cmap='gray')
plt.show()

# In[ ]:

##
'''
    test_img  = test_1 [:,:,None] *255.0
    similar_img =gan_x[:,:,None] *255.0
    # test_img = cv2.resize(test_img,(224,224)) /255.0
    # similar_img = cv2.resize(similar_img,(224,224))/255.0

    np_residual = test_img.reshape(224,224,1) - similar_img.reshape(224,224,1)
    np_residual = (np_residual + 2)/2

    np_residual = (255*np_residual).astype(np.uint8)
    # original_x = (test_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    #　similar_x = (similar_img.reshape(28,28,1)*127.5+127.5).astype(np.uint8)
    original_x = (test_img.reshape(224,224,1)*255.0).astype(np.uint8)
    similar_x = (similar_img.reshape(224,224,1)*255.0).astype(np.uint8)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_COOL) #cv2.COLORMAP_JET COLORMAP_COOL
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)
    qurey = original_x
    print(qurey.shape)
    pred = similar_x
    diff = show
'''
##
i=0
def diff_ori_gan(test_1,gan_x,i):
    ori=np.reshape(test_1[i:i+1], (224, 224,1))

    sim=np.reshape(gan_x[i:i+1], (224, 224,1))
    np_residual =ori-sim

    np_residual = (np_residual + 2)/4
    np_residual=np_residual.astype(np.uint8)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_COOL)
    #print("ori shape",ori.shape)
    original_x_color = cv2.cvtColor(ori.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #print("residual_color.shape",residual_color.shape)

    show = cv2.addWeighted(original_x_color, 0.3,residual_color, 0.7, 0.)
    plt.title("diff")
    plt.imshow(show.astype(np.uint8), cmap='gray')
    if i == 1:
        plt.show()
    return ori,sim, show

def polt_re(qurey,pred,diff,count):
    fig=plt.figure()
    plt.subplot(1,3,1)
    plt.title('query image')
    plt.imshow(qurey.reshape(224,224), cmap=plt.cm.gray)# .reshape(224,224)
    plt.subplot(1,3,2)
    plt.title('generated similar image')
    plt.imshow(pred.reshape(224,224), cmap=plt.cm.gray) #.reshape(224,224)
    plt.subplot(1,3,3)
    plt.title('anomaly detection')
    plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
    fig.savefig('D:\\harden\gan\\GAN-main\\Ganomaly_re\\out\\0727\\{}.png'.format(count),dpi=fig.dpi)
for k in range(len(test_1)):
    qurey,pred,diff=diff_ori_gan(test_1,gan_x,k)
    polt_re(qurey,pred,diff,count=k)


# In[ ]:
from sklearn.metrics import confusion_matrix, roc_curve, auc
# Plot the ROC curve of the test results
def plt_auc(y_test_label,predict_y):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, _ = roc_curve(y_test_label, predict_y)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = {}'.format(roc_auc))

    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
try:
    plt_auc(test_2,all_intscore)
except:
    pass



# In[ ]:

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
    cnf_matrix = metrics.confusion_matrix(test_2,all_intscore)#y_test_label
    target_names = ['OK', 'NG']
    # plot_confusion_matrix(predict_y)
    plot_confusion_matrix(cnf_matrix, classes=target_names)
    # Calculate the sensitivity and specificity
    TP = confusion_matrix(test_2,all_intscore, labels=[1, 0])[0, 0]
    FP = confusion_matrix(test_2,all_intscore, labels=[1, 0])[1, 0]
    FN = confusion_matrix(test_2,all_intscore, labels=[1, 0])[0, 1]
    TN = confusion_matrix(test_2,all_intscore, labels=[1, 0])[1, 1]
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

import pandas as pd
name_list = score
height_list = test_2
df = pd.DataFrame((zip(name_list, height_list)), columns = ['score', 'labe'])
print(df)
df.to_csv("./ganomaly_7271.csv")

import dist 
train_mse = []
test_mse = []
for j in range(len(test_1)):
    mse1,mse2=dist.main_diff(test_1[j],gan_x[j])
    train_mse.append(mse1)
    test_mse.append(mse2)

df2 = pd.DataFrame((zip(train_mse, test_mse)), columns = ['train_mse', 'test_mse'])
df.to_csv("./ganomaly_mse7271.csv")