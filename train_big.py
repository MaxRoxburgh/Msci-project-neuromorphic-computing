import os
import time
import argparse
import pickle

import keras.backend as K 
from keras.optimizers import Adamax
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
import GetAllData as GAD
import numpy as np
from sklearn.model_selection import train_test_split

import plot as pl

# check if code running on GPU
from tensorflow.python.client import device_lib
print('-'*20)
print('List local devices:')
print(device_lib.list_local_devices())
print('\nList GPU devices:')
print(tf.config.list_physical_devices('GPU'))
print('\nRunning on cuda:', tf.test.is_built_with_cuda())


def main(model, model_num, pad=False, ep=80):
    
    ###########################################################################
    # getting data
    
    print('\nLoading Data...')
    x_train, y_train = GAD.get_everything()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=77) # LUKA motherfuckin doncic
    n = x_val.shape[0]
    x_test, y_test = x_train[:n], y_train[:n]
    np.save("x_test_data.npy", x_test)
    np.save("y_test_data.npy", y_test) # save it for later
    x_train, y_train = x_train[n:], y_train[n:]

    
    
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    
    ###########################################################################
    # model building
    
    print('\nModel building ...')
    
    IMG_SHAPE = x_train.shape[1:]
    RESULT_SHAPE = y_train.shape[1:]

    
    encoder, decoder = model(IMG_SHAPE, RESULT_SHAPE)
    inp = encoder
    reconstruction = decoder

    def accuracy(y_true, y_pred):

        # Consider values within 5% difference as correct
        within_threshold = K.abs(y_true - y_pred) <= 0.05 * y_true

        # Consider difference within 0.5 as correct
        # 0.5 is the resolution of model
        special_case = K.abs(y_true - y_pred) <= 0.5

        # Combine conditions to get correct predictions
        correct = within_threshold | special_case

        return K.mean(correct)
    
    try:
        autoencoder = Model(inp, reconstruction)
        loss = 'mse'
        opt = Adamax(learning_rate = 0.00005, epsilon=1e-8)
        autoencoder.compile(optimizer=opt, loss=loss, metrics=[accuracy])
    except:
        print("failed to load optimiser")
        autoencoder = Model(inp, reconstruction)
        loss = 'mse'
        autoencoder.compile(optimizer="adamax", loss=loss, metrics=[accuracy])

    print(autoencoder.summary())
    
    
    ###########################################################################
    # model training

    print('\nTraining ...\n')
    start_time = time.time()
    
    # train branch 1: large batch
    batch_size = 8
    ep = 350
    print("batch size:", batch_size)
    print("epochs:", ep)
    history = autoencoder.fit(x=x_train, y=y_train, epochs=ep, verbose=2, validation_data=[x_val, y_val],
                              batch_size=batch_size)  
    
    end_time = time.time()-start_time
    print('\nTraining took', end_time, 's')
    print('which is', end_time/3600, 'hrs')
    
    ###########################################################################
    # save model

    dir_folder = os.getcwd().replace("\\", "/") + f"/big_train/model_{model_num}_bs8_ep500_mse"
    os.makedirs(dir_folder, exist_ok=True)
    pl.losshistory(history.history, dir_folder, True)
    with open(dir_folder + '/losshistory-dict','wb') as file:
        pickle.dump(history.history, file)

    autoencoder.save(dir_folder + f"/model_test_{model_num}.keras")

    print("\nSaved model to disk")

    print('-'*20)

import models as m
main(m.unet_model_3, "3")