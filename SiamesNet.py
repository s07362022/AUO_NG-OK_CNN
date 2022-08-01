import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras import backend as K
import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

num_classes = 2 
epochs = 1000 #500

###
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

###
 
 
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
 
 
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
 
 
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
 
 
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
 
 
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)  # add 
    x = Dense(64, activation='relu')(x) # add 
    x = Dropout(0.2)(x)  # add 
    x = Dense(32, activation='relu')(x) # add 
    return Model(input, x)
 
 
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
 
 
def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
 
'''
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]
'''
IMAGE_SIZE = 300
import cv2
import os
D = "D:\\harden\\dataset\\H1_water_de\\OK2\\"   #####path
E = "D:\\harden\\dataset\\H1_water_de\\water_cut_NG\\"
images1 = []
labels1 = []
dir_counts = 0
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
def d (D=D,images=images1,labels=labels1):
    vou=0
    for i in os.listdir(D):
        
        try:
            #print(D+i)
            img1 = cv2.imread(D+i)
            #print(D+i)
            #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            #print(0)
            labels.append(dir_counts+1) # not +1
        except:
            print("error")
        vou +=1
        if vou >=2000:
            break
    print("A already read")
    return(images,labels)
d(D,images1,labels1) # OK sample to label: 1
test_00b = images1[0]
def e (E=E,images=images1,labels=labels1):
    BC = 0
    for i in os.listdir(E):
        try:
            img2 = cv2.imread(E+i)
            #img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
            img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img2)
            labels.append(dir_counts)
        except:
            print("error")
        BC = BC+1
        if BC == 2000:
            break
    print("B already read")
    return(images,labels)
e(E,images1,labels1) # NG sample to label: 0
print("LAB",labels1)
from sklearn.model_selection import train_test_split
label = np.array(labels1)
X_train,X_test,y_train,y_test =  train_test_split(images1, label,test_size=0.1,random_state=42 )#
x_train = np.array(X_train, dtype=np.float32)
x_test = np.array(X_test, dtype=np.float32)
x_train = x_train/255.0
x_test  =  x_test/255.0
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
tr_y=tr_y.astype(float)
##### create dataset focus here!!!!!!!!
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
print("digit_indices",digit_indices)
te_pairs, te_y = create_pairs(x_test, digit_indices)
##### create dataset focus here!!!!!!!!
te_y=te_y.astype(float)
# network definition
input_shape=(300,300,3)
base_network = create_base_network(input_shape)
 
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

print("len len(te_pairs)",len(te_pairs))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)
 
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
 
model = Model([input_a, input_b], distance)


# train
if args.mode== 'train':
    rms = RMSprop()
    adam =Adam(lr=0.00008)
    model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=16,#128,
              epochs=epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

if args.mode== 'test':
    model.load_weights('weights/siamesnet1.h5')
    print("model load True ")

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)
 
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print("shape te_pairs",te_pairs.shape)

# save model 
model.save('weights/siamesnet1.h5')
print("model save")


import cv2
from tensorflow.keras.models import load_model
# predict data with model
def predict(model_path, image_path1, image_path2, target_size,i,lable_te):
    #saved_model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss})
    saved_model = model_path
    # image1 = cv2.imread(image_path1)
    # image2 = cv2.imread(image_path2)
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.resize(image1, target_size)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.resize(image2, target_size)  # <class 'numpy.ndarray'>
    image1 = image_path1
    image2 = image_path2
    # print(image2.shape)  # (80, 80)
    # print(image2)
    data1 = np.array([image1], dtype='float') #/ 255.0 / 255.0
    data2 = np.array([image2], dtype='float') #/ 255.0 / 255.0
    #print(data1.shape, data2.shape)  # (1, 80, 80) (1, 80, 80)
    pairs = np.array([data1, data2])
    #print(pairs.shape)  # (2, 80, 80)
    
    y_pred = saved_model.predict([data1, data2])
    print(y_pred)
    # print(y_pred)  # [[4.1023154]]
    pred = y_pred.ravel() < 0.5
    print("第 %d 個 pred: " % i ,pred)  # 相似程度
    y_true = [1]  # 1表示同一類, 0表示不同類
    if pred == y_true:
        print("視同一類")
    else:
        print("不是同一類")
    ##############################################
    #try:
    #if i <=10 :
        #plt.subplots(211)
        #plt.title('pre: %s , label: %s , te_label: %s ' %(str(pred),str(te_y[i]),"0"), fontsize=6)
        #plt.imshow(data1)
        #plt.subplots(212)
        #plt.title('pre: %s , label: %s , te_label: %s ' %(str(pred),str(te_y[i]),"1"), fontsize=6)
        #plt.imshow(data2)
        #plt.show()
    #except:
        #pass 
    ##############################################ssss
    return pred

pre=[]
for i in range(len(te_pairs)):
    score=predict(model,te_pairs[i,1],te_pairs[i, 0],target_size=(300,300),i=i,lable_te=te_y)
    pre.append(score)
#plt.show()



#img1=(test_00b ).astype(int)
#img2=(te_pairs[-3, 0] *255).astype(int)
#print(te_pairs.shape)

#plt.imshow(img1)
#plt.show()

#plt.imshow(img2)
#plt.show()
#####
def plot_image(image,labels,prediction,idx,num=20):  
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
plot_image(te_pairs[:,1],te_y[:],pre[:],idx=0,num=len(te_pairs))

plot_image(te_pairs[:,0],te_y[:],pre[:],idx=0,num=len(te_pairs))



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
    plt_auc(te_y[:],pre[:])
except:
    pass


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
    cnf_matrix = metrics.confusion_matrix(te_y[:],pre[:])#y_test_label
    
    target_names = ['NG', 'OK']
    # plot_confusion_matrix(predict_y)
    plot_confusion_matrix(cnf_matrix, classes=target_names)
    # Calculate the sensitivity and specificity
    TP = confusion_matrix(te_y[:],pre[:], labels=[1, 0])[0, 0]
    FP = confusion_matrix(te_y[:],pre[:], labels=[1, 0])[1, 0]
    FN = confusion_matrix(te_y[:],pre[:], labels=[1, 0])[0, 1]
    TN = confusion_matrix(te_y[:],pre[:], labels=[1, 0])[1, 1]
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


def polt_re(qurey,pred,diff,count):
    fig=plt.figure()
    plt.subplot(1,3,1)
    plt.title('paris 0')
    plt.imshow(qurey.reshape(300,300,3), cmap=plt.cm.gray)# .reshape(64,64)
    plt.subplot(1,3,2)
    plt.title('paris 1')
    plt.imshow(pred.reshape(300,300,3), cmap=plt.cm.gray) #.reshape(64,64)
    plt.subplot(1,3,3)
    plt.title('anomaly detection')
    plt.imshow(diff)
    fig.savefig('D:\\harden\\siameNet\\0727\\output{}.png'.format(k),dpi=fig.dpi)

for k in range(len(te_y)):
    if te_y[k] ==1.0:
        te_y[k]=True #[1]
        print("ok")
    #print(pre[k][0]==1.0)
    if te_y[k] != pre[k][0]:
        # print(te_pairs[k,0])
        print("te_y[k]=",te_y[k],"pre[k]:",pre[k][0])
        diff=(te_pairs[k,0]*255.0)-(te_pairs[k,1]*255.0)
        original_x_color = (te_pairs[k,0])*255.0
        diff = cv2.applyColorMap((diff).astype(np.uint8), cv2.COLORMAP_JET)#*255.0
        diff = cv2.addWeighted(original_x_color.astype(np.uint8), 0.3, diff.astype(np.uint8), 0.7, 0.)
        polt_re(qurey=(te_pairs[k,0]*255.0).astype(np.uint8),pred=(te_pairs[k,1]*255.0).astype(np.uint8),diff=diff,count=0)
        cv2.imwrite("D:\\harden\\siameNet\\0727\\pair0_{}.png".format(k),(te_pairs[k,0]*255.0).astype(np.uint8))
        cv2.imwrite("D:\\harden\\siameNet\\0727\\pair1_{}.png".format(k),(te_pairs[k,1]*255.0).astype(np.uint8))