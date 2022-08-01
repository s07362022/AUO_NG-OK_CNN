import os
import sys
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

##set input image size
input_shape = (64,64,3)
IMAGE_SIZE = 64
#resize
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w, _ = image.shape
    
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

#load_data
A = "D:\\harden\\dataset\\H1_water_de\\OK\\"
#fall1 = "C:\\Users\\User.DESKTOP-IIINHE5\\Desktop\\fall1_img"
B = "D:\\harden\\dataset\\H1_water_de\\NG\\"
images = []
labels = []
dir_counts = 0
for i in os.listdir(A):
    img1 = cv2.imread(A+i)
    # img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
    img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
    images.append(img1)
    labels.append(dir_counts)
    #a = np.array(images,dtype=np.float32)
    #print(a.shape)

for i in os.listdir(B):
    img2 = cv2.imread(B+i)
    # img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
    img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
    images.append(img2)
    labels.append(dir_counts+1)
    #a = np.array(images,dtype=np.float32)
    #print(a.shape)

label = np.array(labels)
#########Train/Test############
X_train_img,X_test_img,y_train_label,y_test_label =  train_test_split(images, label,test_size=0.2,random_state=2 )#
X_train = np.array(X_train_img, dtype=np.float32)
X_test = np.array(X_test_img, dtype=np.float32)
#print("X_train.shape",X_train.shape)
x_train_std = X_train/255.0
x_test_std  =  X_test/255.0
y_trainOneHot = np_utils.to_categorical(y_train_label)
y_testOneHot = np_utils.to_categorical(y_test_label)
print("x_test_std.shape",x_test_std.shape)
print("y_test_label",y_test_label.shape)
################################


# Set the augmentation parameters and fit the training data
############### change here #################
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.0,
    zoom_range=0.0,
    fill_mode="constant",
    cval=0
)
############### change here #################
datagen.fit(x_train_std)



#Model
'''
from tensorflow.keras.applications import MobileNet
base_model = MobileNet(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000
)
'''
from tensorflow.keras.applications import EfficientNetV2M
base_model=EfficientNetV2M(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=input_shape,#None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'#,
    #include_preprocessing=True
)



x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions) #inputs=base_model.input

print("summary: ", model.summary())

# Compile the model
model.compile(
      optimizer=keras.optimizers.Adam(3e-4),
      loss="binary_crossentropy",
      metrics=["accuracy"],
)
#summary
#model.summary()

# Set the epochs and batch size, then train the model
############### change here #################
epochs = 5
batch_size = 4
############### change here #################

history = model.fit(
    datagen.flow(x_train_std, y_trainOneHot, batch_size=batch_size),
    steps_per_epoch=len(X_train)/batch_size,
    epochs=epochs,
    validation_data=(x_test_std, y_testOneHot)
)

history_dict = history.history
print(history_dict.keys())
import matplotlib.pyplot as plt
# Plot loss and accuracy
def plt_loss(history):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left') 
    #plt.show()
    # summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left') 
    plt.show()
plt_loss(history)

#Predict
predict_y = model.predict(x_test_std)



#confusion_matrix
predict_y[predict_y >= 0.5] = 1
predict_y[predict_y < 0.5] = 0
print(confusion_matrix(y_testOneHot.argmax(axis=1), predict_y.argmax(axis=1), labels=[1, 0]))

y=y_testOneHot.argmax(axis=1)
p_y=predict_y.argmax(axis=1)
# Calculate the sensitivity and specificity
TP = confusion_matrix(y, p_y, labels=[1, 0])[0, 0]
FP = confusion_matrix(y, p_y, labels=[1, 0])[1, 0]
FN = confusion_matrix(y, p_y, labels=[1, 0])[0, 1]
TN = confusion_matrix(y, p_y, labels=[1, 0])[1, 1]
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
plt_auc(y,predict_y.argmax(axis=1))

def plot_image(image,labels,prediction,idx,num=10):  
    fig = plt.gcf() 
    fig.set_size_inches(12, 14) 
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
plot_image(x_test_std,y_test_label,predict_y.argmax(axis=1),idx=10)