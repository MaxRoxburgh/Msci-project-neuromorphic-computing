# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:50:59 2024

@author: Maxwell
"""

from keras.models import load_model
import tensorflow as tf
# from tensorflow.utils import plot_model
from IPython.display import Image

# model = load_model("M:\\MSci_neuromorphic-dt-main\\history\\model_updated_size.keras")
model = load_model("M:\\BSc_neuromorphic-dt-main\\Mask-Training-Code\\tools\\pretrained_model\\model-BOHB-MAE.keras")
print(model.summary())
# tf.keras.utils.plot_mode(model, to_file='model_plot.png')
# Image(filename='model_plot.png')
#%%
import datasets as ds

path = ""
data = ds.load(path)
