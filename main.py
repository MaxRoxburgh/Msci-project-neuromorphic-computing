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

# check if code running on GPU
from tensorflow.python.client import device_lib
print('-'*20)
print('List local devices:')
print(device_lib.list_local_devices())
print('\nList GPU devices:')
print(tf.config.list_physical_devices('GPU'))
print('\nRunning on cuda:', tf.test.is_built_with_cuda())


def main(model, model_num, pad=False):#args):
    
    ###########################################################################
    # getting data
    print('\nLoading Data...')
    x_train, x_val, y_train, y_val = GAD.get_all_data(pad=True, size=138)
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
    # print(IMG_SHAPE,RESULT_SHAPE)
    # print("prod r_shape", np.prod(RESULT_SHAPE))
    # input("wait")
    
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
    loss = 'mae'
    autoencoder.compile(optimizer='adamax', loss=loss, metrics=[accuracy])
    print(autoencoder.summary())
    
    
    ###########################################################################
    # model training
    
    print('\nTraining ...\n')
    input("press something to continue")
    start_time = time.time()
    
    # train
    batch_size = 64
    print("batch size:", batch_size)
    # steps_per_epoch = number of test data/ batch_size
    # steps_per_epoch = 200
    ep = 150
    print("epochs:", ep)
    history = autoencoder.fit(x=x_train, y=y_train, epochs=ep, verbose=0, validation_data=[x_val, y_val],
                              batch_size=batch_size)#, steps_per_epoch=steps_per_epoch)

    # plot and save loss history
    dir_folder = os.getcwd().replace("\\", "/") + f"/history/model_{model_num}"
    os.makedirs(dir_folder, exist_ok=True)
    pl.losshistory(history.history, dir_folder, True)#args.plot_show)
    with open(dir_folder + '/losshistory-dict','wb') as file:
        pickle.dump(history.history, file)
        
    end_time = time.time()-start_time
    print('\nTraining took', end_time, 's')
    print('which is', end_time/3600, 'hrs')
    
    ###########################################################################
    # save model
    
    autoencoder.save(dir_folder + f"/model_updated_size{model_num}.keras")

    print("\nSaved model to disk")

    print('-'*20)
    # print("Arguments:\n", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model-number', dest='model_num', required=True, type=int,
        help='Model number for identification')
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
else:
    args.model_num = 6
    from models import unet_model_6
    model = unet_model_6  
    
main(model, model_num=f"{args.model_num}", pad=True)    
    
    