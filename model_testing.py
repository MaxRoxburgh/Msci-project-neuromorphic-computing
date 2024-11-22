# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:50:59 2024

@author: Maxwell
"""

from keras.models import load_model
import tensorflow as tf
# from tensorflow.keras.utils import plot_model
import keras
from IPython.display import Image

# model = load_model("M:\\MSci_neuromorphic-dt-main\\history\\model_updated_size.keras")
model = load_model("M:/MSci_neuromorphic-dt-main/history/model_3/model_updated_size3.keras")
print(model.summary())
keras.utils.plot_model(model, to_file='model_plot.png')
# Image(filename='model_plot.png')
#%%
import datasets as ds

path = "C:/Users/Maxwell/Imperial College London/complex nanophotonics - PH - 20241101_sanity checks/data/ham95_shuffled_gTrue_rot1_/DATA_20241101_Toriel120+_VoNetW350L05D150_ham95_full_gTrue_rot1_0001.ds"
spectrum_data = ds.load(path)
path_2 = "C:/Users/Maxwell/Imperial College London/complex nanophotonics - PH - 20241101_sanity checks/source_images/ham95_shuffled.ds"
input_data = ds.load(path_2)

from tools import add_padding_to_images

test_images = add_padding_to_images(input_data.raw, 138)
#%%
predicted_spectrums = model.predict(test_images[:100])


#%%
n = len(spectrum_data.raw)
spectrum_data = spectrum_data.raw[:,10:74,65:193].reshape(n,64,64,2).mean(axis=-1).reshape(n,64,64,1)


#%%
import matplotlib.pyplot as plt
for i in range(0,100, 5):
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(test_images[i])
    ax[1].imshow(spectrum_data[i])
    ax[2].imshow(predicted_spectrums[i])
    print(f"\n\nfor index {i}, max: pred = {max(predicted_spectrums[i].flatten())} experimental = {max(spectrum_data[i].flatten())}")
    print(f"for index {i}, min: pred = {min(predicted_spectrums[i].flatten())} experimental = {min(spectrum_data[i].flatten())}")
    plt.show()

#%%
import numpy as np
av_pred = [np.average(i.flatten()) for i in predicted_spectrums]
av_expr = [np.average(i.flatten()) for i in spectrum_data[:100]]

x = [i+1 for i in range(100)]
plt.plot(x, av_pred)
plt.plot(x, av_expr)

#%%
fig, ax = plt.subplots(3,5)
x = [i for i in range(64)]
for axss in ax:
    for axs in axss:
        axs.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False) 

for n, i in enumerate([33,44,55]):
    ax[n,0].plot(x, predicted_spectrums[i,25,:])
    ax[n,0].plot(x, spectrum_data[i,25,:])
    ax[n,1].plot(x, predicted_spectrums[i,33,:])
    ax[n,1].plot(x, spectrum_data[i,33,:])
    ax[n,2].plot(x, predicted_spectrums[i,40,:])
    ax[n,2].plot(x, spectrum_data[i,40,:])
    ax[n,3].plot(x, predicted_spectrums[i,48,:])
    ax[n,3].plot(x, spectrum_data[i,48,:])
    ax[n,4].plot(x, predicted_spectrums[i,56,:])
    ax[n,4].plot(x, spectrum_data[i,56,:])
    
    
    
#%% cifar10 gray 
    
path = "C:/Users/Maxwell/Imperial College London/complex nanophotonics - PH - 20241101_sanity checks/data/cifar10_gray_gTrue_rot0_/DATA_20241101_Toriel120+_VoNetW350L05D150_cifar10_gray_gTrue_rot0_0001.ds"
spectrum_data = ds.load(path).raw
#%%
path = "C:/Users/Maxwell/Imperial College London/complex nanophotonics - PH - 20241101_sanity checks/source_images/cifar10_gray.ds"
source_data = ds.load(path).raw
source_data = np.rot90(source_data, -1, axes=(1,2))    
#%%
plt.imshow(source_data[0])

#%%
    
from tools import resample_image_sp, resample_image_by_scale

rescaled = resample_image_sp(source_data[:5], 95)    
rescaled_mine = np.array([resample_image_by_scale(i, 95) for i in source_data[:5]])
#%%
n = len(spectrum_data)
spectrum_data = spectrum_data[:,10:74,65:193].reshape(n,64,64,2).mean(axis=-1).reshape(n,64,64,1)
#%%

i=1

fig, axs = plt.subplots(1,3)
axs[0].imshow(rescaled_mine[i])
axs[0].set_title("origonal")
axs[1].imshow(rescaled[i])  
axs[1].set_title("sp")
axs[2].imshow(rescaled_mine[i])
axs[2].set_title("mine")
plt.show()

#%%

preds = model.predict(add_padding_to_images(rescaled_mine, 138, 255))

#%%

for i in range(5):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(preds[i])
    ax[1].imshow(spectrum_data[i])

    ax[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False) 
    ax[1].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False) 
    