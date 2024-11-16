"""
Choped and stolen from BA project

This doc contains:
    optimised network
    Workflow for training model
    
"""

import sys
import os
import io
import gc
import time
import argparse
import pickle
import json

import keras
import numpy as np
import keras.backend as K 
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import GetAllData as GAD

import plot as pl

# check if code running on GPU
from tensorflow.python.client import device_lib
print('-'*20)
print('List local devices:')
print(device_lib.list_local_devices())
print('\nList GPU devices:')
print(tf.config.list_physical_devices('GPU'))
print('\nRunning on cuda:', tf.test.is_built_with_cuda())

# def unet(IMG_SHAPE, RESULT_SHAPE):
    


def unet_optimised(IMG_SHAPE, RESULT_SHAPE):
    """
    unet optimised by BA students and collated into a single object with no if statements
    """    
    
    config = {'activation': 'relu', 'conca': 'twice', 'decov': 'Conv2DTranspose', 'drop': 0.0655936065918576,
              'drop_mid': 0.21194937479704087, 'f1': 0, 'f2': 3, 'f3': 3, 'k1': 10, 'k2': 25, 'k3': 30,
              'ker_init': 'glorot_normal', 'pool_type': 'average'}
    
    input_net = Input(IMG_SHAPE)
    print("input_net shape:", input_net.shape)
    f_list = [(3, 3, 3, 3, 3, 3), (5, 5, 2, 2, 3, 3), (5, 5, 3, 3, 2, 3), (3, 3, 2, 2, 4, 3),
               (2, 4, 2, 4, 4, 2), (2, 2, 4, 3, 3, 3)]
    
    # set parameters
    f = f_list[config["f1"]]  # select which tuple from f_list
    k1 = config["k1"]
    activation = config["activation"]
    ker_init = config["ker_init"]
    drop = config["drop"]
    pool_type = config["pool_type"]
    k2 = config["k2"]
    k3 = config["k3"]
    drop_mid = config["drop_mid"]
    conca = config["conca"]
    decov = config["decov"]
    f2 = config["f2"]
    f3 = config["f3"]
    
    # The encoder
    print("Encoder Shape")
    enc1 = Conv2D(k1, (f[0], f[0]), activation=activation, kernel_initializer=ker_init)(input_net)
    print("\n\nenc1 shape:", enc1.shape)
    d1 = Dropout(drop)(enc1)
    print("d1 shape:", d1.shape)
    enc2 = Conv2D(k1, (f[1], f[1]), activation=activation, kernel_initializer=ker_init)(d1)
    print("enc2 shape:", enc2.shape)
    enc3 = AveragePooling2D((2, 2))(enc2)
    print("enc3 shape:", enc3.shape)
    enc4 = Conv2D(k2, (f[2], f[2]), activation=activation, kernel_initializer=ker_init)(enc3)
    print("enc4 shape:", enc4.shape)
    d2 = Dropout(drop)(enc4)
    print("d2 shape:", d2.shape)
    enc5 = Conv2D(k2, (f[3], f[3]), activation=activation, kernel_initializer=ker_init)(d2)
    print("enc5 shape:", enc5.shape)
    enc6 = AveragePooling2D((2, 2))(enc5)
    print("enc6 shape:", enc6.shape)
    enc7 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(enc6)
    print("enc7 shape:", enc7.shape)
    d_mid = Dropout(drop_mid)(enc7)
    print("dmid shape:", d_mid.shape)
    enc8 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(d_mid)
    print("enc8 shape:", enc8.shape)
    
    print("\n")
    # input("\n [PAUSE] look at encoder shapes")
    
    # The decoder
    dec0 = Conv2DTranspose(k2, (2, 2), strides=(2, 2))(enc8)

    try:
        merge1 = concatenate([dec0, enc5])
    except ValueError:  # crop tensor if shapes do not match
        shape_de = K.int_shape(dec0)[1]
        shape_en = K.int_shape(enc5)[1]
        diff = shape_de - shape_en

        if diff > 0:
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            dec0 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(dec0)

        elif diff < 0:
            diff = -int(1 * diff)
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            enc5 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(enc5)

        merge1 = concatenate([dec0, enc5])

    dec1 = Conv2D(k2, (f[4], f[4]), activation=activation, kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(k2, (f[5], f[5]), activation=activation, kernel_initializer=ker_init, padding='same')(dec1)
    dec3 = Conv2DTranspose(k1, (2, 2), strides=(2, 2))(dec2)

    try:
        merge2 = concatenate([dec3, enc4])
    except:  # crop tensor if shapes do not match
        shape_de = K.int_shape(dec3)[1]
        shape_en = K.int_shape(enc4)[1]
        diff = shape_de - shape_en

        if diff > 0:
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            dec3 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(dec3)
        elif diff < 0:
            diff = -int(1 * diff)
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            enc4 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(enc4)

        merge2 = concatenate([dec3, enc4])



    dec4 = Conv2D(k1, (f2, f2), activation=activation, kernel_initializer=ker_init)(merge2)
    print("dec4 shape:", dec4.shape)
    dec5 = Conv2D(k1, (f3, f3), activation=activation, kernel_initializer=ker_init)(dec4)
    print("dec5 shape:", dec5.shape)
    # dec6 = Conv2DTranspose(k1, (3, 3), strides=(3, 3))(dec5)
    dec6 = dec5
    print("dec6 shape:", dec6.shape)
    dec7 = Flatten()(dec6)
    print("dec7 shape:", dec7.shape)
    dec8 = Dense(np.prod(RESULT_SHAPE))(dec7)
    print("dec8 shape:", dec8.shape)
    output_net = Reshape(RESULT_SHAPE)(dec8)  # 64, 64, 1
    print("output_net")
    return input_net, output_net

def unet_optimised_2(IMG_SHAPE, RESULT_SHAPE):
    """
    unet optimised by BA students and collated into a single object with no if statements
    """    
    
    config = {'activation': 'relu', 'conca': 'twice', 'decov': 'Conv2DTranspose', 'drop': 0.0655936065918576,
              'drop_mid': 0.21194937479704087, 'f1': 0, 'f2': 3, 'f3': 3, 'k1': 18, 'k2': 150, 'k3': 150,
              'ker_init': 'glorot_normal', 'pool_type': 'average'}
    
    input_net = Input(IMG_SHAPE)
    print("input_net shape:", input_net.shape)
    f_list = [(3, 3, 3, 3, 3, 3), (5, 5, 2, 2, 3, 3), (5, 5, 3, 3, 2, 3), (3, 3, 2, 2, 4, 3),
               (2, 4, 2, 4, 4, 2), (2, 2, 4, 3, 3, 3)]
    
    # set parameters
    f = f_list[config["f1"]]  # select which tuple from f_list
    k1 = config["k1"]
    activation = config["activation"]
    ker_init = config["ker_init"]
    drop = config["drop"]
    pool_type = config["pool_type"]
    k2 = config["k2"]
    k3 = config["k3"]
    drop_mid = config["drop_mid"]
    conca = config["conca"]
    decov = config["decov"]
    f2 = config["f2"]
    f3 = config["f3"]
    pool_size_1 = 4
    pool_size_2 = 2
    
    # The encoder
    print("Encoder Shape")
    enc1 = Conv2D(k1, (f[0], f[0]), activation=activation, kernel_initializer=ker_init)(input_net)
    print("\n\nenc1 shape:", enc1.shape)
    d1 = Dropout(drop)(enc1)
    print("d1 shape:", d1.shape)
    enc2 = Conv2D(k1, (f[1], f[1]), activation=activation, kernel_initializer=ker_init)(d1)
    print("enc2 shape:", enc2.shape)
    enc3 = AveragePooling2D((pool_size_1, pool_size_1))(enc2)
    print("enc3 shape:", enc3.shape)
    enc4 = Conv2D(k2, (f[2], f[2]), activation=activation, kernel_initializer=ker_init)(enc3)
    print("enc4 shape:", enc4.shape)
    d2 = Dropout(drop)(enc4)
    print("d2 shape:", d2.shape)
    enc5 = Conv2D(k2, (f[3], f[3]), activation=activation, kernel_initializer=ker_init)(d2)
    print("enc5 shape:", enc5.shape)
    enc6 = AveragePooling2D((pool_size_2, pool_size_2))(enc5)
    print("enc6 shape:", enc6.shape)
    enc7 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(enc6)
    print("enc7 shape:", enc7.shape)
    d_mid = Dropout(drop_mid)(enc7)
    print("dmid shape:", d_mid.shape)
    enc8 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(d_mid)
    print("enc8 shape:", enc8.shape)
    
    print("\n")
    # input("\n [PAUSE] look at encoder shapes")
    
    # The decoder
    dec0 = Conv2DTranspose(k2, (2, 2), strides=(2, 2))(enc8)

    try:
        merge1 = concatenate([dec0, enc5])
    except ValueError:  # crop tensor if shapes do not match
        shape_de = K.int_shape(dec0)[1]
        shape_en = K.int_shape(enc5)[1]
        diff = shape_de - shape_en

        if diff > 0:
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            dec0 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(dec0)

        elif diff < 0:
            diff = -int(1 * diff)
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            enc5 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(enc5)

        merge1 = concatenate([dec0, enc5])

    dec1 = Conv2D(k2, (f[4], f[4]), activation=activation, kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(k2, (f[5], f[5]), activation=activation, kernel_initializer=ker_init, padding='same')(dec1)
    dec3 = Conv2DTranspose(k1, (2, 2), strides=(2, 2))(dec2)

    try:
        merge2 = concatenate([dec3, enc4])
        print("dec and enc4 shapes", dec3.shape, enc4.shape)
    except:  # crop tensor if shapes do not match
        shape_de = K.int_shape(dec3)[1]
        shape_en = K.int_shape(enc4)[1]
        diff = shape_de - shape_en
        print("dec and enc4 shapes", dec3.shape, enc4.shape)
        if diff > 0:
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            dec3 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(dec3)
        elif diff < 0:
            diff = -int(1 * diff)
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            enc4 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(enc4)

        merge2 = concatenate([dec3, enc4])



    dec4 = Conv2D(k1, (f2, f2), activation=activation, kernel_initializer=ker_init)(merge2)
    print("dec4 shape:", dec4.shape)
    dec5 = Conv2D(k1, (f3, f3), activation=activation, kernel_initializer=ker_init)(dec4)
    print("dec5 shape:", dec5.shape)
    # dec6 = Conv2DTranspose(k1, (3, 3), strides=(3, 3))(dec5)
    dec6 = dec5
    print("dec6 shape:", dec6.shape)
    dec7 = Flatten()(dec6)
    print("dec7 shape:", dec7.shape)
    dec8 = Dense(np.prod(RESULT_SHAPE))(dec7)
    print("dec8 shape:", dec8.shape)
    output_net = Reshape(RESULT_SHAPE)(dec8)  # 64, 64, 1
    print("output_net")
    return input_net, output_net


def main():#args):
    
    ###########################################################################
    # getting data
    print('\nLoading Data...')
    x_train, x_val, y_train, y_val = GAD.get_all_data()
    x_train, x_val, y_train, y_val = tf.stack(x_train), tf.stack(x_val), tf.stack(y_train), tf.stack(y_val)
    
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    
    # plt.imshow(x_train[0])
    # plt.show()
    # plt.imshow(y_train[0])
    # plt.show()
    
    # input("\nTaking a break for a while...\n")
    
    ###########################################################################
    # model building
    print('\nModel building ...')
    
    IMG_SHAPE = x_train.shape[1:]
    RESULT_SHAPE = y_train.shape[1:]
    print(IMG_SHAPE,RESULT_SHAPE)
    print("prod r_shape", np.prod(RESULT_SHAPE))
    # input("wait")
    
    encoder, decoder = unet_optimised_2(IMG_SHAPE, RESULT_SHAPE)
    inp = encoder
    reconstruction = decoder
    # input("another break...")
    def accuracy(y_true, y_pred):

        # Consider values within 5% difference as correct
        within_threshold = K.abs(y_true - y_pred) <= 0.05 * y_true

        # Consider difference within 0.5 as correct
        # 0.5 is the resolution of model
        special_case = K.abs(y_true - y_pred) <= 0.5

        # Combine conditions to get correct predictions
        correct = within_threshold | special_case

        return K.mean(correct)
    
    autoencoder = Model(inp, reconstruction)
    loss = 'mse'
    autoencoder.compile(optimizer='adamax', loss=loss, metrics=[accuracy])
    print(autoencoder.summary())
    
    
    ###########################################################################
    # model training
    
    print('\nTraining ...\n')
    input("press something to continue")
    start_time = time.time()
    
    # train
    batch_size = 128
    # steps_per_epoch = number of test data/ batch_size
    # steps_per_epoch = 200
    ep = 150
    history = autoencoder.fit(x=x_train, y=y_train, epochs=ep, verbose=2, validation_data=[x_val, y_val],
                              batch_size=batch_size)#, steps_per_epoch=steps_per_epoch)

    # plot and save loss history
    dir_folder = os.getcwd() + "\\history\\model_2"
    pl.losshistory(history.history, dir_folder, True)#args.plot_show)
    with open(dir_folder + '\\losshistory-dict','wb') as file:
        pickle.dump(history.history, file)
        
    end_time = time.time()-start_time
    print('\nTraining took', end_time, 's')
    print('which is', end_time/3600, 'hrs')
    
    ###########################################################################
    # save model
    
    autoencoder.save(dir_folder + "\\model_updated_size2.keras")

    print("\nSaved model to disk")

    print('-'*20)
    # print("Arguments:\n", args)

main()    
    
    