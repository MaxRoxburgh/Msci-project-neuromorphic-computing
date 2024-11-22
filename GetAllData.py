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
from tools import add_padding_to_images

def get_all_data(pad=False, size=95, home=False):
    """
    Returns
    -------
    x_train, x_test, y_train, y_test
    
    """

    # Take relevant paths
    if home:
        path = "C:/Users/Maxwell/Imperial College London/complex nanophotonics - PH - 20241101_sanity checks"
    else:
        path = os.getcwd().replace("\\", "/") + "/twin_data"
    
    spectrum_paths_rot0 = glob.glob(path + "/data/isic12_95_gTrue_rot0_/*[0-9].ds")
    spectrum_paths_rot0 = [i.replace("\\", "/") for i in spectrum_paths_rot0]
    spectrum_paths_rot1 = glob.glob(path + "/data/isic12_95_gTrue_rot1_/*[0-9].ds")
    spectrum_paths_rot1 = [i.replace("\\", "/") for i in spectrum_paths_rot1]
    input_path = path + "/source_images/isic12_95.ds"
    
    # All input data
    inputs = ds.load(input_path).raw
    
    rot = 0
    input_rot0 = np.rot90(inputs, -1+rot, axes=(1, 2))
    
    rot = 1
    input_rot1 = np.rot90(inputs, -1+rot, axes=(1, 2))
    
    pad_value = 157
    
    x_total = np.concatenate((input_rot0, input_rot1), axis=0)
    if pad:
        x_total = add_padding_to_images(x_total, size, pad_value)
        # import matplotlib.pyplot as plt
        # plt.imshow(x_total[0])
        # plt.show()
        # plt.imshow(x_total[1])
        # plt.show()
        # plt.imshow(x_total[5000])
        # plt.show()
        # plt.imshow(x_total[3])
        # plt.show()
        # plt.imshow(x_total[4])
        # plt.show()
        # plt.imshow(x_total[5])
        # plt.show()
        # return
        
    # All output data
    y_total = ds.load(spectrum_paths_rot0[0]).raw
    
    for i in spectrum_paths_rot0[1:]:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
    
    for i in spectrum_paths_rot1:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
        
    y_total = y_total[:,10:74,65:193].reshape(len(x_total),64,64,2).mean(axis=-1).reshape(len(x_total),64,64,1)
    x_total = x_total.reshape(len(x_total),size,size,1)
    
    print("total no. items:", y_total.shape[0])
    return train_test_split(x_total, y_total, test_size=0.1, random_state=42)

# get_all_data(True, 138)
# print()