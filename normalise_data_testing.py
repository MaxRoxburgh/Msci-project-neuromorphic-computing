# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:32:33 2024

@author: Maxwell
"""
import datasets as ds
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

def data_average():
    """
    Returns
    -------
    x_train, x_test, y_train, y_test
    
    """
    import matplotlib.pyplot as plt
    # Take relevant paths
    # path = os.getcwd() + "\\twin_data"
    path = "C:\\Users\\Maxwell\\Imperial College London\\complex nanophotonics - PH - 20241101_sanity checks"
    
    spectrum_paths_rot0 = glob.glob(path + "\\data\\isic12_95_gTrue_rot0_\\*[0-9].ds")
    spectrum_paths_rot1 = glob.glob(path + "\\data\\isic12_95_gTrue_rot1_\\*[0-9].ds")
    input_path = path + "\\source_images\\isic12_95.ds"
    
    # All input data
    inputs = ds.load(input_path).raw
    
    rot = 0
    input_rot0 = np.rot90(inputs, -1+rot, axes=(1, 2))
    plt.imshow(input_rot0[0])
    plt.show()
    
    rot = 1
    input_rot1 = np.rot90(inputs, -1+rot, axes=(1, 2))
    plt.imshow(input_rot1[0])
    plt.show()
    
    print(np.mode([np.mode(i.flatten()) for i in input_rot0]))
    print(np.mode([np.mode(i.flatten()) for i in input_rot1]))
    
    x_total = np.concatenate((input_rot0, input_rot1), axis=0)
    
    # All output data
    y_total = ds.load(spectrum_paths_rot0[0]).raw
    
    for i in spectrum_paths_rot0[1:]:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
    
    y_averages_rot0 = [np.average(i.flatten()) for i in y_total]
    
    y_total_2 = ds.load(spectrum_paths_rot1[0]).raw
    for i in spectrum_paths_rot1[1:]:
        y_temp = ds.load(i).raw
        y_total_2 = np.concatenate((y_total_2, y_temp))
        
    y_averages_rot1 = [np.average(i.flatten()) for i in y_total_2]
    
    y = y_averages_rot0[:500]
    plt.plot([i for i in range(len(y))], y)
    y = y_averages_rot1[:500]
    plt.plot([i for i in range(len(y))], y)
    plt.show()
    
    diff_percentage = [(1- i/j) * 100 for i, j in zip(y_averages_rot0, y_averages_rot1)]
    y = diff_percentage[:500]
    plt.plot([i for i in range(len(y))], y)
    plt.show()
    
    plt.imshow(y_total[0])
    print("spectrum average 1:", np.average(y_total[0].flatten()))
    print(min(y_total[0].flatten()), max(y_total[0].flatten()))
    plt.show()
    plt.imshow(y_total_2[0])
    print("spectrum average 2:", np.average(y_total_2[0].flatten()))
    print(min(y_total_2[0].flatten()), max(y_total_2[0].flatten()))
    plt.show()
    
    plt.imshow(y_total[3])
    print("spectrum average 1:", np.average(y_total[3].flatten()))
    print(min(y_total[3].flatten()), max(y_total[3].flatten()))
    plt.show()
    plt.imshow(y_total_2[3])
    print("spectrum average 2:", np.average(y_total_2[3].flatten()))
    print(min(y_total_2[3].flatten()), max(y_total_2[3].flatten()))
    plt.show()
    
    print("average for data 1 spectrums:", np.average(y_averages_rot0))
    print("average for data 2 spectrums:", np.average(y_averages_rot1))
    
    print("min for first five:", [min(i.flatten()) for i in y_total[:5]])
    print("max for first five:", [max(i.flatten()) for i in y_total[:5]])
    
    # y_total = y_total[:,10:74,65:193].reshape(len(x_total),64,64,2).mean(axis=-1).reshape(len(x_total),64,64,1)
    # x_total = x_total.reshape(len(x_total),95,95,1)
    
        
    # return train_test_split(x_total, y_total, test_size=0.1, random_state=42)

data_average()
    