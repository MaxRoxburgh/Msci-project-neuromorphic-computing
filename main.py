"""
Choped and stolen from BA project

This doc contains:
    optimised network
    Workflow for training model
    
"""

import os
import time
import argparse
import pickle

import keras.backend as K 
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
import GetAllData as GAD

import plot as pl
from tools import TrendBasedEarlyStopping

# check if code running on GPU
from tensorflow.python.client import device_lib
print('-'*20)
print('List local devices:')
print(device_lib.list_local_devices())
print('\nList GPU devices:')
print(tf.config.list_physical_devices('GPU'))
print('\nRunning on cuda:', tf.test.is_built_with_cuda())


def main(model, model_num, pad=False, ep=200):#args):
    
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

    
    ###########################################################################
    # model building
    
    print('\nModel building ...')
    
    IMG_SHAPE = x_train.shape[1:]
    RESULT_SHAPE = y_train.shape[1:]

    
    encoder, decoder = model(IMG_SHAPE, RESULT_SHAPE)
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
    
    early_stopping = TrendBasedEarlyStopping(
        monitor_train='loss',   # Metric to monitor for training
        monitor_val='val_loss', # Metric to monitor for validation
        patience=8              # Number of epochs to evaluate the trend
    )
        
    print('\nTraining ...\n')
    # input("press something to continue")
    start_time = time.time()
    
    # train branch 1: large batch
    batch_size = 128
    print("batch size:", batch_size)
    print("epochs:", ep)
    history = autoencoder.fit(x=x_train, y=y_train, epochs=ep, verbose=2, validation_data=[x_val, y_val],
                              batch_size=batch_size, callbacks=[early_stopping])
    
    # plot and save loss history branch 1
    dir_folder = os.getcwd().replace("\\", "/") + f"/history_bigdata/model_{model_num}/branch1"
    os.makedirs(dir_folder, exist_ok=True)
    pl.losshistory(history.history, dir_folder, True)#args.plot_show)
    with open(dir_folder + '/losshistory-dict','wb') as file:
        pickle.dump(history.history, file)
    
    print("\nLarge batch complete; fine tuning...\n")    
    
    batch_size = 8
    print("batch size:", batch_size)
    print("epochs:", ep)
    
    history = autoencoder.fit(x=x_train, y=y_train, epochs=ep, verbose=2, validation_data=[x_val, y_val],
                              batch_size=batch_size, callbacks=[early_stopping])
    
    # plot and save loss history branch 2
    dir_folder = os.getcwd().replace("\\", "/") + f"/history_bigdata/model_{model_num}/branch2"
    os.makedirs(dir_folder, exist_ok=True)
    pl.losshistory(history.history, dir_folder, True)#args.plot_show)
    with open(dir_folder + '/losshistory-dict','wb') as file:
        pickle.dump(history.history, file)
    
    end_time = time.time()-start_time
    print('\nTraining took', end_time, 's')
    print('which is', end_time/3600, 'hrs')
    
    ###########################################################################
    # save model
    dir_folder = os.getcwd().replace("\\", "/") + f"/history_bigdata/model_{model_num}"
    autoencoder.save(dir_folder + f"/model_updated_size{model_num}.keras")

    print("\nSaved model to disk")

    print('-'*20)
    # print("Arguments:\n", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model-number', dest='model_num', required=True, type=int,
        help='Model number for identification')
    parser.add_argument(
        '-ep', '--epochs', dest='ep', required=False, type=int,
        help='Number of epochs for training')
    args = parser.parse_args()

if args.model_num == 1:
    from models import unet_model_1
    model = unet_model_1
elif args.model_num == 2:
    from models import unet_model_2
    model = unet_model_2
elif args.model_num == 3:
    from models import unet_model_3
    model = unet_model_3
elif args.model_num == 4:
    from models import unet_model_4
    model = unet_model_4
elif args.model_num == 5:
    from models import unet_model_5
    model = unet_model_5    
elif args.model_num == 7:
    from models import unet_model_7
    model = unet_model_7
else:
    args.model_num = 6
    from models import unet_model_6
    model = unet_model_6  

if not args.ep:
    main(model, model_num=f"{args.model_num}", pad=True)    
else:
    main(model, model_num=f"{args.model_num}", pad=True, ep=args.ep)    
    
    