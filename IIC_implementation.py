# -*- coding: utf-8 -*-

from google.colab import drive

drive.mount("/content/drive")

#----- PACKAGES -----
import random
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import os
from scipy.optimize import linear_sum_assignment as lsa
from imgaug import augmenters as iaa
from math import ceil

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from tensorflow.keras.applications.vgg16 import VGG16
from scipy.optimize import linear_sum_assignment

from keras.datasets import mnist

#Model parameters
EPOCHS = 501
STEPS = 80
BATCH_SIZE=720

INIT_LR = 1e-4
MAIN_OUTPUT_UNITS=10

CP_HEAD_ITERATIONS=10
CP_IIC_MODEL_ITERATIONS=50

#----- DATASET IMPORT -----
IMG_SIZE=24

[x_train,y_train],[x_test,y_test]=mnist.load_data()
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

classes=[0,1,2,3,4,5,6,7,8,9]

#TRANSFORMATIONS

rotate=tf.keras.Sequential([layers.experimental.preprocessing.RandomRotation(factor=15, fill_mode='nearest', interpolation='bilinear')])

flip=tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip(mode="horizontal")])

resize = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(height=IMG_SIZE,width=IMG_SIZE)])

def crop_transf(batch,crop_fraction):
  uncropped_size = IMG_SIZE-int((crop_fraction*IMG_SIZE)//100)
  crop = tf.keras.Sequential([layers.experimental.preprocessing.CenterCrop(height=uncropped_size,width=uncropped_size),layers.experimental.preprocessing.Resizing(height=IMG_SIZE,width=IMG_SIZE)])
  batch=crop(batch)
  return batch

def hsv_transf(image):

  val1=random.uniform(0.5,1.5)
  val2=random.uniform(0.5,1.5)
  val3=random.uniform(0.5,1.5)
  
  image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  h,s,v = cv2.split(image)
  h = h*val1
  s = s*val2
  v = v*val3
  cv2.merge([h,s,v],image)
  cv2.cvtColor(image, cv2.COLOR_HSV2RGB,dst=image)

  return image


sobelFilter = K.variable([[[[3,3]], [[0.,10]],[[-3,3]]],
                          [[[10,0]], [[0.,0]],[[-10,0]]],
                          [[[3,-3]], [[0.,-10]],[[-3,-3]]]])


laplacianFilter = K.variable([[[[0 ]], [[-1 ]],[[0 ]]],
                      [[[-1 ]], [[4 ]],[[-1 ]]],
                      [[[0 ]], [[0 ]],[[0 ]]]])

def expandedLaplacian(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return laplacianFilter * inputChannels

def expandedSobel(inputTensor):

    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above

    return sobelFilter * inputChannels

def laplacian_func(batch,filt=laplacianFilter):

    #get the sobel filter repeated for each input channel
    #filt = expandedLaplacian(batch)
    batch=tf.image.rgb_to_grayscale(batch)
    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    laplacian = K.depthwise_conv2d(batch,filt)
    resize = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(height=IMG_SIZE,width=IMG_SIZE)])
    batch=resize(batch)
    return batch

    #now you just apply the mse:
    return laplacian
def converter(x):
    #x has shape (batch, width, height, channels)
    return (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])

def sobel_func(batch ):
    
    batch=tf.image.rgb_to_grayscale(batch)

    filt=expandedSobel(batch)
    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    batch = K.depthwise_conv2d(batch,filt)
    
    resize = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(height=IMG_SIZE,width=IMG_SIZE)])
    batch=resize(batch)
    batch=batch[:,:,:,0]
    batch=tf.expand_dims(batch,axis=3)
    return batch

def data_generator(batch_size=BATCH_SIZE):
  while True:
        
        z=[]
        z1=[]
        #Select 21+1 indxs  
        random_images_indx=random.sample(range(60000),int(batch_size/3))

        #Triplicate each index in order to match the 3 transformations per image
        triplicated_image_indx=[ i for i in random_images_indx for r in range(3) ]


        #21 cropped + 21 multiplied + 21 flipped images
        transf_samples =x_train[triplicated_image_indx]

        for i in range(len(transf_samples)):
          rand_val=random.sample(range(3),1)
          if rand_val==0:
               transf_samples[i]= crop_transf(transf_samples[i],25)
          if rand_val==1:
               transf_samples[i]= flip(transf_samples[i])
          if rand_val==2:
               transf_samples[i]= rotate(transf_samples[i])
            
        z=np.array(resize(x_train[triplicated_image_indx]))
        z1=np.array(resize(transf_samples))

        #shuffler = list(np.random.permutation(batch_size))
        #z = np.array([z[shuffler[i],:,:,:] for i in shuffler])
        #z1 =  np.array([z1[shuffler[i],:,:,:] for i in shuffler])
        
        yield ([z,z1],np.zeros((batch_size,1)).astype("float32"))

def main_loss(y_true,y_pred,batch_size=BATCH_SIZE,lamb=1):

    k=MAIN_OUTPUT_UNITS
    y_pred=tf.squeeze(y_pred)

    #Divide the outputs that have been concatenated 
    phi1=tf.squeeze(y_pred[:,0:MAIN_OUTPUT_UNITS])
    phi2=tf.squeeze(y_pred[:,MAIN_OUTPUT_UNITS:])

    P= tf.reduce_sum(tf.expand_dims(phi1, 2) * tf.expand_dims(phi2, 1), 0)

    #Symmetrize P matrix
    P=tf.add(P,tf.transpose(P))/2

    #Add eps value in order to avoid 0 values in P
    P=tf.clip_by_value(P,clip_value_min=1e-6,clip_value_max=1e9)   

    P/=tf.reduce_sum(P)

    pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(P, axis=0), (k, 1)), (k, k))
    pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(P, axis=1), (1, k)), (k, k))
    loss = -tf.reduce_sum(P * (tf.math.log(P) - lamb * tf.math.log(pi) - lamb * tf.math.log(pj)))

    return loss

def networkB(input,filters):

    X=input

    F1,F2,F3,F4 = filters

    #BLOCK 1
    X = Conv2D(filters = F1, kernel_size=(5,5), strides =(1,1),padding = 'same', kernel_initializer = 'random_normal', use_bias=False)(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding="same")(X)

    #BLOCK 2
    X = Conv2D(filters = F2, kernel_size=(3,3), strides =(1,1),padding = 'same', kernel_initializer = 'random_normal', use_bias=False)(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding="same")(X)

    #BLOCK 3
    X = Conv2D(filters = F3, kernel_size=(3,3), strides =(1,1),padding = 'same', kernel_initializer = 'random_normal', use_bias=False)(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding="same")(X)

    #BLOCK 4
    X = Conv2D(filters = F4, kernel_size=(3,3), strides =(1,1),padding = 'same', kernel_initializer = 'random_normal', use_bias=False)(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)


    #BLOCK 5
    X=Flatten()(X)
    return X

#Input head model
input1=Input([IMG_SIZE,IMG_SIZE,1])
input2=Input([IMG_SIZE,IMG_SIZE,1])
#input3=Input([IMG_SIZE,IMG_SIZE,1])
#input4=Input([IMG_SIZE,IMG_SIZE,1])
#input5=Input([IMG_SIZE,IMG_SIZE,1])


base_output_1=networkB(input1,[64,128,256,512])
#base_output_2=networkB(input2,[64,128,256,512])
#base_output_3=networkB(input3,[64,128,256,512])
#base_output_4=networkB(input4,[64,128,256,512])
#base_output_5=networkB(input5,[64,128,256,512])

#Define the 5 heads output
output_1 =Dense(MAIN_OUTPUT_UNITS,activation="softmax", kernel_initializer='random_normal', bias_initializer='zeros')(base_output_1)
#output_2 =Dense(MAIN_OUTPUT_UNITS,activation="softmax", kernel_initializer='random_normal', bias_initializer='zeros')(base_output_2)
#output_3 =Dense(MAIN_OUTPUT_UNITS,activation="softmax", kernel_initializer='random_normal', bias_initializer='zeros')(base_output_3)
#output_4 =Dense(MAIN_OUTPUT_UNITS,activation="softmax", kernel_initializer='random_normal', bias_initializer='zeros')(base_output_4)
#output_5 =Dense(MAIN_OUTPUT_UNITS,activation="softmax", kernel_initializer='random_normal', bias_initializer='zeros')(base_output_5)



#Define the 5 models related to the 5 heads
model_1 = Model(inputs=input1, outputs=output_1 , name="head_model_1")
#model_2 = Model(inputs=input2, outputs=output_2 , name="head_model_2")
#model_3 = Model(inputs=input3, outputs=output_3 , name="head_model_3")
#model_4 = Model(inputs=input4, outputs=output_4 , name="head_model_4")
#model_5 = Model(inputs=input5, outputs=output_5 , name="head_model_5")



input_1=Input([IMG_SIZE,IMG_SIZE,1])
input_2=Input([IMG_SIZE,IMG_SIZE,1])

#Define the two models output
model_1_out=Concatenate(name="model_1_output")([model_1(input_1),model_1(input_2)])
#model_2_out=Concatenate(name="model_2_output")([model_2(input_1),model_2(input_2)])
#model_3_out=Concatenate(name="model_3_output")([model_3(input_1),model_3(input_2)])
#model_4_out=Concatenate(name="model_4_output")([model_4(input_1),model_4(input_2)])
#model_5_out=Concatenate(name="model_5_output")([model_5(input_1),model_5(input_2)])


#Define the entire model
IIC_model=Model(inputs=[input_1,input_2],outputs=[model_1_out])

#Loss weights callback function definition
class  CustomCallback(Callback):

    def on_epoch_begin(self, epoch, logs=None):
      if epoch%CP_HEAD_ITERATIONS==0:

        checkpoint_m1="/content/drive/MyDrive/Colab Notebooks/IIC_implementation/cp_head_"+str(epoch)+".h5"
        model_1.save_weights(checkpoint_m1)


      if epoch%CP_IIC_MODEL_ITERATIONS==0:
        checkpoint_IIC="/content/drive/MyDrive/Colab Notebooks/IIC_implementation/cp_IIC_"+str(epoch)+".h5"
        IIC_model.save_weights(checkpoint_IIC)

losses = [main_loss]
lossWeights = (1)
opt = Adam(lr=INIT_LR  )

#%load_ext tensorboard
# Clear any logs from previous runs
#!rm -rf ./logs/ 

IIC_model.compile(optimizer=opt, loss=losses , loss_weights=lossWeights, run_eagerly=False)
#IIC_model.load_weights("/content/drive/MyDrive/Colab Notebooks/IIC_implementation/checkpoint_MAIN_AVG_130.h5")

#log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#IIC_model.fit(data_generator(),steps_per_epoch=STEPS,epochs=EPOCHS,verbose=1,callbacks=[CustomCallback(),tensorboard_callback])
IIC_model.fit(data_generator(),steps_per_epoch=STEPS,epochs=EPOCHS,verbose=1,callbacks=[CustomCallback()])

def  CustomMetric(model,x_true,y_true, num_classes):

    x_true=resize(x_true)
  
    #x_true=tf.image.rgb_to_grayscale(x_true)
    softmax_predictions=model.predict(x_true)
    predictions=[np.argmax(x) for x in softmax_predictions]
  
    # initialize count matrix
    cnt_mtx = np.zeros([num_classes, num_classes])

    # fill in matrix
    for i in range(len(y_true)):
        cnt_mtx[int(predictions[i]), int(y_true[i])] += 1

    # find optimal permutation
    row_ind, col_ind = linear_sum_assignment(-cnt_mtx)
    #print(row_ind)
    #print(col_ind)

    # compute error
    error = 1 - cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()

    # print results
    #print('Classification error = {:.4f}'.format(error))

    return error

# -------------SHOW RESULTS--------------
cp_list=[]
eval_range=range(0,EPOCHS,CP_HEAD_ITERATIONS)
for i in eval_range:
  try:
    model_1.load_weights("/content/drive/MyDrive/Colab Notebooks/IIC_implementation/cp_head_"+str(i)+".h5")
    cp_list.append(CustomMetric(model_1,x_test,y_test,CP_HEAD_ITERATIONS))
  except:
    continue
    
plt.plot(eval_range, cp_list) 
plt.show()

#%tensorboard --logdir logs/fit --port=8006

def convolutional_block(X, filters):
    
    # Retrieve Filters
    F1 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = F1, kernel_size=(3,3), strides = (2,2) , padding = 'same', kernel_initializer = 'random_normal')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)

    # Second component of main path (≈2 lines)
    X = Conv2D(filters = F1, kernel_size = (3,3), strides = (2,2) , padding = 'same', kernel_initializer = 'random_normal')(X)
    X = BatchNormalization(axis = -1 )(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F1, kernel_size = (3,3), strides = (4,4), padding = 'same', kernel_initializer = 'random_normal')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def identity_block(X, filters):
    
    # Retrieve Filters
    F1, F2 , F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = F1, kernel_size=(3, 3), strides =(1,1),padding = 'same', kernel_initializer = 'random_normal')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Activation('relu')(X)

    # Second component of main path (≈2 lines)
    X = Conv2D(filters = F2, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer = 'random_normal')(X)
    X = BatchNormalization(axis = -1 )(X)
    X = Activation('relu')(X)

    # third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer = 'random_normal')(X)
    X = BatchNormalization(axis = -1 )(X)
    X = Activation('relu')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
