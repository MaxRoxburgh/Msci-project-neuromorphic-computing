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

def get_all_data():
    """
    Returns
    -------
    x_train, x_test, y_train, y_test
    
    """

    # Take relevant paths
    # path = os.getcwd() + "\\twin_data"
    path = "C:\\Users\\Maxwell\\Imperial College London\\complex nanophotonics - PH - 20241101_sanity checks"
    
    spectrum_paths_rot0 = glob.glob(path + "\\data\\isic12_95_gTrue_rot0_\\*[0-9].ds")
    spectrum_paths_rot1 = glob.glob(path + "\\data\\isic12_95_gTrue_rot1_\\*[0-9].ds")
    input_path = path + "\\source_images\\isic12_95.ds"
    
    # All input data
    input_rot0 = ds.load(input_path).raw
    rot = 1
    input_rot1 = np.rot90(input_rot0, -1+rot, axes=(1, 2))
    
    x_total = np.concatenate((input_rot0, input_rot1), axis=0)
    
    # All output data
    y_total = ds.load(spectrum_paths_rot0[0]).raw
    
    for i in spectrum_paths_rot0[1:]:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
    
    for i in spectrum_paths_rot1:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
        
    y_total = y_total[:,10:74,65:193].reshape(len(x_total),64,64,2).mean(axis=-1).reshape(len(x_total),64,64,1)
    x_total = x_total.reshape(len(x_total),95,95,1)
    
        
    return train_test_split(x_total, y_total, test_size=0.1, random_state=42)
    
    
