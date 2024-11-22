import GetAllData as gad
import os

#########################################
###### accounting for random seeds ######
import numpy as np
import random
import tensorflow as tf
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
#########################################

def remove_ticks(axs):
    for ax in axs:
        ax.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False) 


x_t, x_v, y_t, y_v = gad.get_all_data(True, 138, False)
x_t, x_v, y_t, y_v = tf.stack(x_t), tf.stack(x_v), tf.stack(y_t), tf.stack(y_v)
# (6649, 138, 138, 1) (739, 138, 138, 1) (6649, 64, 64, 1) (739, 64, 64, 1)

# test indexes of images
t_index = [i+10 for i in range(0, 6650, 250)][:25] 
v_index = [i for i in range(0,740,27)][:25] 

# make folder to store images
dir_folder = os.getcwd().replace("\\", "/") + "/images_test"
os.makedirs(dir_folder, exist_ok=True)

# plots and saving
import matplotlib.pyplot as plt

fig, axs = plt.subplots(5,5)
axs = axs.flatten()
remove_ticks(axs)
for i, j in enumerate(t_index):
    axs[i].imshow(x_t[j])
fig.savefig(dir_folder+"/x_t_test.png")
 
fig, axs = plt.subplots(5,5)
axs = axs.flatten()
remove_ticks(axs)
for i, j in enumerate(t_index):
    axs[i].imshow(y_t[j])
fig.savefig(dir_folder+"/y_t_test.png")

fig, axs = plt.subplots(5,5)
axs = axs.flatten()
remove_ticks(axs)
for i, j in enumerate(v_index):
    axs[i].imshow(x_v[j])
fig.savefig(dir_folder+"/x_v_test.png")
 
fig, axs = plt.subplots(5,5)
axs = axs.flatten()
remove_ticks(axs)
for i, j in enumerate(v_index):
    axs[i].imshow(y_v[j])
fig.savefig(dir_folder+"/y_v_test.png")




