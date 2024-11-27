import datasets as ds
import numpy as np
import glob
from tools import resample_imporved, multiplied, add_padding_to_images, spectrum_retrival

def mnist(path):

    power = 310

    input_path = path + "/source_images/mnist_full_rotated.ds"
    spectrum_paths_MNIST = glob.glob(path + "/data/mnist_rot0_/*[0-9].ds")
    spectrum_paths_MNIST = sorted([i.replace("\\", "/") for i in spectrum_paths_MNIST])

    x_mnist = 255 - ds.load(input_path).raw
    x_mnist = multiplied(x_mnist, power)
    x_mnist = resample_imporved(x_mnist, 133)

    y_mnist = ds.load(spectrum_paths_MNIST[0]).raw
    for i in spectrum_paths_MNIST[1:]:
        y_temp = ds.load(i).raw
        y_mnist = np.concatenate((y_mnist, y_temp), axis=0)
    y_mnist = y_mnist[:,10:74,65:193].reshape(len(y_mnist),64,64,2).mean(axis=-1).reshape(len(y_mnist),64,64)  

    return x_mnist, y_mnist

def cifar10_gray(path):
    # Cifar10_gray
    
    spectrum_paths_cifar = glob.glob(path + "/data/cifar10_gray_gTrue_rot0_/*[0-9].ds")
    spectrum_paths_cifar = sorted([i.replace("\\", "/") for i in spectrum_paths_cifar])
    input_path = path + "/source_images/cifar10_gray.ds"
    
    # All input data cifar10 gray
    pad_value = 255
    power = 400
    rot = 0
    
    input_cifar_gray = ds.load(input_path).raw
    input_cifar_gray = multiplied(input_cifar_gray, power)
    input_cifar_gray = np.rot90(input_cifar_gray, -1+rot, axes=(1, 2))
    # input_cifar_gray = np.array([resample_image_by_scale(i, 96) for i in input_cifar_gray])
    input_cifar_gray = resample_imporved(input_cifar_gray, 96)
    input_cifar_gray = add_padding_to_images(input_cifar_gray, 133, pad_value*power).reshape(len(input_cifar_gray), 133, 133)
    
    # All output data
    y_cifar = ds.load(spectrum_paths_cifar[0]).raw
    
    for i in spectrum_paths_cifar[1:]:
        y_temp = ds.load(i).raw
        y_cifar = np.concatenate((y_cifar, y_temp))
    
    y_cifar = y_cifar[:,10:74,65:193].reshape(len(y_cifar),64,64,2).mean(axis=-1).reshape(len(y_cifar),64,64)

    return input_cifar_gray, y_cifar

def isic12_95(path):

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
    input_rot0 = multiplied(input_rot0, power)
    rot = 1
    input_rot1 = np.rot90(inputs, -1+rot, axes=(1, 2))
    input_rot1 = multiplied(input_rot1, power)
    
    x_total = np.concatenate((input_rot0, input_rot1), axis=0)

    x_total = add_padding_to_images(x_total, 133, pad_value*power)

        
    # All output data
    y_total = ds.load(spectrum_paths_rot0[0]).raw
    
    for i in spectrum_paths_rot0[1:]:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
    
    for i in spectrum_paths_rot1:
        y_temp = ds.load(i).raw
        y_total = np.concatenate((y_total, y_temp))
        
    y_total = y_total[:,10:74,65:193].reshape(len(y_total),64,64,2).mean(axis=-1).reshape(len(y_total),64,64)
    x_total = x_total.reshape(len(x_total),133,133)

    return x_total, y_total

def cifar10_colour(path):
    # Cifar10_colour
    
    spectrum_paths_cifar_r = glob.glob(path + "/data/cifar10_r_gTrue_rot0_/*[0-9].ds")
    spectrum_paths_cifar_g = glob.glob(path + "/data/cifar10_g_gTrue_rot0_/*[0-9].ds")
    spectrum_paths_cifar_b = glob.glob(path + "/data/cifar10_b_gTrue_rot0_/*[0-9].ds")
    spectrum_paths_cifar_r = sorted([i.replace("\\", "/") for i in spectrum_paths_cifar_r])
    spectrum_paths_cifar_g = sorted([i.replace("\\", "/") for i in spectrum_paths_cifar_g])
    spectrum_paths_cifar_b = sorted([i.replace("\\", "/") for i in spectrum_paths_cifar_b])
    input_path = path + "/source_images/cifar10.ds"

    
    # All input data cifar10 gray
    pad_value = 255
    power = 400
    rot = 0
    
    input_cifar = ds.load(input_path).raw
    input_cifar = multiplied(input_cifar, power)
    input_cifar = np.rot90(input_cifar, -1+rot, axes=(1, 2))
    input_cifar_r = input_cifar[:,:,:,0]
    input_cifar_g = input_cifar[:,:,:,1]
    input_cifar_b = input_cifar[:,:,:,2]

    input_cifar_r = resample_imporved(input_cifar_r, 96)
    input_cifar_g = resample_imporved(input_cifar_g, 96)
    input_cifar_b = resample_imporved(input_cifar_b, 96)

    input_cifar_r = add_padding_to_images(input_cifar_r, 133, pad_value*power).reshape(len(input_cifar_r), 133, 133)
    input_cifar_g = add_padding_to_images(input_cifar_g, 133, pad_value*power).reshape(len(input_cifar_g), 133, 133)
    input_cifar_b = add_padding_to_images(input_cifar_b, 133, pad_value*power).reshape(len(input_cifar_b), 133, 133)
    
    # All output data

    y_cifar_r = spectrum_retrival(spectrum_paths_cifar_r)
    y_cifar_g = spectrum_retrival(spectrum_paths_cifar_g)
    y_cifar_b = spectrum_retrival(spectrum_paths_cifar_b)

    return np.concatenate( (np.concatenate((input_cifar_r,input_cifar_g)),input_cifar_b) ), np.concatenate((np.concatenate((y_cifar_r,y_cifar_g)),y_cifar_b))

def breakhist(path):
    
    bh_specrtum_rot0_paths =  glob.glob(path + "/data/breakhist_div2_unquart_gTrue_rot0_/*[0-9].ds")
    bh_spectrum_rot1_paths = glob.glob(path + "/data/breakhist_div2_unquart_gTrue_rot1_/*[0-9].ds")
    bh_specrtum_rot0_paths = sorted([i.replace("\\", "/") for i in bh_specrtum_rot0_paths])
    bh_spectrum_rot1_paths = sorted([i.replace("\\", "/") for i in bh_spectrum_rot1_paths])
    input_path = path + "/source_images/breakhist_div2_unquart.ds"

    # input images
    pad_level = 213
    power = 319

    input_breakhist = ds.load(input_path).raw
    input_breakhist = resample_imporved(input_breakhist, 96)
    input_breakhist = multiplied(input_breakhist, power)
    rot = 0
    input_breakhist_rot0 = np.rot90(input_breakhist, -1+rot, axes=(1, 2))
    rot = 1
    input_breakhist_rot1 = np.rot90(input_breakhist, -1+rot, axes=(1, 2))

    input_breakhist_comb = np.concatenate((input_breakhist_rot0, input_breakhist_rot1))
    input_breakhist_comb = add_padding_to_images(input_breakhist_comb, 133, pad_level*power)

    # output specrta

    y_breakhist_rot0 =  spectrum_retrival(bh_specrtum_rot0_paths)
    y_breakhist_rot1 =  spectrum_retrival(bh_spectrum_rot1_paths)

    return input_breakhist_comb, np.concatenate((y_breakhist_rot0, y_breakhist_rot1))






