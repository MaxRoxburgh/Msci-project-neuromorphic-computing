# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:21:59 2024

@author: Maxwell
"""

import datasets as ds
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tools import add_padding_to_images, resample_image_by_scale

def get_all_data(home=False):
    """
    essentially formats all the data exactly how it would have been put into the experimental set up
    ML architecture was built on 138 length as I was initally given wrong numbers by PhD...
    actual should be 133 so now everything has an extra layer of 5 pixels so it's (138,138) which are all 0
    -------
    Returns
    -------
    x_train, x_test, y_train, y_test
    
    """

    # Take relevant paths
    if home:
        path = "C:/Users/Maxwell/Imperial College London/complex nanophotonics - PH - 20241101_sanity checks"
    else:
        path = os.getcwd().replace("\\", "/") + "/twin_data"
        
    # isic_95:
    
    spectrum_paths_rot0 = glob.glob(path + "/data/isic12_95_gTrue_rot0_/*[0-9].ds")
    spectrum_paths_rot0 = sorted([i.replace("\\", "/") for i in spectrum_paths_rot0])
    spectrum_paths_rot1 = glob.glob(path + "/data/isic12_95_gTrue_rot1_/*[0-9].ds")
    spectrum_paths_rot1 = sorted([i.replace("\\", "/") for i in spectrum_paths_rot1])
    input_path = path + "/source_images/isic12_95.ds"
        
    # All input data isic_95
    pad_value = 157
    power = 310
    
    inputs = ds.load(input_path).raw
    rot = 0
    input_rot0 = np.rot90(inputs, -1+rot, axes=(1, 2))
    rot = 1
    input_rot1 = np.rot90(inputs, -1+rot, axes=(1, 2))
    
    
    x_total = np.concatenate((input_rot0, input_rot1), axis=0)
    x_total *= power

    x_total = add_padding_to_images(x_total, 133, pad_value*power)

        
    # All output data
    y_total = ds.load(spectrum_paths_rot0[0]).raw
    
    for i in spectrum_paths_rot0[1:]:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
    
    for i in spectrum_paths_rot1:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
        
    y_total = y_total[:,10:74,65:193].reshape(len(x_total),64,64,2).mean(axis=-1).reshape(len(x_total),64,64)
    x_total = x_total.reshape(len(x_total),133,133)
    
    # Cifar10_gray
    
    spectrum_paths_cifar = glob.glob(path + "/data/cifar10_gray_gTrue_rot0_/*[0-9].ds")
    spectrum_paths_cifar = sorted([i.replace("\\", "/") for i in spectrum_paths_cifar])
    input_path = path + "/source_images/cifar10_gray.ds"
    
    # All input data cifar10 gray
    pad_value = 255
    power = 400
    rot = 0
    
    input_cifar_gray = ds.load(input_path).raw*power
    input_cifar_gray = np.rot90(input_cifar_gray, -1+rot, axes=(1, 2))
    input_cifar_gray = np.array([resample_image_by_scale(i, 96) for i in input_cifar_gray])
    input_cifar_gray = add_padding_to_images(input_cifar_gray, 133, pad_value*power).reshape(len(input_cifar_gray), 133, 133)
    
    x_total = np.concatenate((x_total, input_cifar_gray), axis=0)
    x_total = add_padding_to_images(x_total, 138).reshape(len(x_total), 138, 138, 1)
    
    # All output data
    y_cifar = ds.load(spectrum_paths_cifar[0]).raw
    
    for i in spectrum_paths_cifar[1:]:
        y_temp = ds.load(i).raw
        y_cifar = np.concatenate((y_cifar, y_temp))
    
    y_cifar = y_cifar[:,10:74,65:193].reshape(len(x_total),64,64,2).mean(axis=-1).reshape(len(x_total),64,64)
    y_total = np.concatenate((y_total, y_cifar), axis=0)
    y_total = y_total.reshape(len(y_total), 64, 64, 1)
        
        
    
    print("total no. items:", y_total.shape[0])
    return train_test_split(x_total, y_total, test_size=0.1, random_state=42)

