from keras.layers import Conv2D, Dropout, AveragePooling2D, concatenate, Cropping2D
import keras.backend as K 
import numpy as np

def encoder_sub_block(input, NumFilter1, NumFilter2, KernelSize1, KernelSize2, DropoutRate, DonwsampleSize, ker_init='random_normal', activation='relu', debug=False, all_encoders=False):
    """
    two convolutional layers seperated by a dropout layer, AveragePooling2D after
    ----------
    PARAMETERS:
    ----------        
    input : ndarray 
        input_net

    NumFilter1 : int 
        number of convolutional filters in the first layer

    NumFilter2 : int
        number of convolutional filters in the second layer

    KernelSize1 : (int, int)
        convolutional kernel size for first convolutional layer

    KernelSize2 : (int, int)
        convolutional kernel size for second convolutional layer

    DropoutRate : float
        dropout between the first and second layer

    DonwsampleSize : (int, int)
        the kernel for the average pooling layer
    
    ker_init : str
        initialisation type of the 

    activation : str
        activation type for convolutional layers
    """

    enc1 = Conv2D(NumFilter1, KernelSize1, activation=activation, kernel_initializer=ker_init)(input) 
    d1 = Dropout(DropoutRate)(enc1)
    enc2 = Conv2D(NumFilter2, KernelSize2, activation=activation, kernel_initializer=ker_init)(d1) 
    enc3 = AveragePooling2D(DonwsampleSize)(enc2) 

    if debug:
        print("Layer 1 shape:", enc1.shape)
        print("Layer 2 shape:", enc2.shape)
        print("Layer 3 shape:", enc3.shape)

    if all_encoders:
        return enc1, enc2, enc3
    else:
        return enc3

def encoder_mid_block(input, NumFilter1, NumFilter2, DropoutMid, ker_init='random_normal', activation='relu', debug=False):
    """
    two convolutional layers with a dropout layer in the middle, mid block for a U-Net
    ----------
    PARAMETERS
    ----------
    input : ndarray 
        input_net

    NumFilter1 : int 
        number of convolutional filters in the first layer

    NumFilter2 : int
        number of convolutional filters in the second layer
    
    DropoutMid : float   
        dropout between the first and second layer
    
    ker_init : str
        initialisation type of the 

    activation : str
        activation type for convolutional layers
    """
    enc1 = Conv2D(NumFilter1, (1, 1), activation=activation, kernel_initializer=ker_init)(input)
    d_mid = Dropout(DropoutMid)(enc1)
    enc2 = Conv2D(NumFilter2, (1, 1), activation=activation, kernel_initializer=ker_init)(d_mid)
    

    if debug:
        print("\nMid block shapes")
        print("enc1 shape:", enc1.shape)
        print("dmid shape:", d_mid.shape)
        print("enc2 shape:", enc2.shape)
        print("")


    return enc2


def crop_and_merge(dec, enc, debug=False):
    """
    if the encoder and decoder layers arn't matching in size, this crops the larger to size
    ----------
    Parameters
    ----------
    dec : ndarray
        decoder layer to be merged
    enc : ndarray
        encoder layer to be merged
    """
    
    shape_de = K.int_shape(dec)[1]
    shape_en = K.int_shape(enc)[1]
    diff = shape_de - shape_en

    if debug:
        print("Crop debug: (before)\n")
        print("decoder layer shape:", dec.shape)
        print("encoder layer shape:", dec.shape)

    if diff > 0:
        diff_bottom = int(round(diff / 2))
        diff_top = int(diff - diff_bottom)
        dec = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(dec)
    elif diff < 0:
        diff = -int(1 * diff)
        diff_bottom = int(round(diff / 2))
        diff_top = int(diff - diff_bottom)
        enc = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(enc)
    
    if debug:
        print("\n\nCrop debug: (after)\n")
        print("decoder layer shape:", dec.shape)
        print("encoder layer shape:", dec.shape)

    return concatenate([dec, enc])

def add_padding_to_images(input_array, new_size, padding_value=0):
    """
    Adds padding to every picture in the input array.
    
    Parameters:
        input_array (numpy.ndarray): The input array of shape (n_images, height, width).
        new_size (int): The desired new width and height of the padded images.
        padding_value (int/float): The value to use for padding. Default is 0.
        
    Returns:
        numpy.ndarray: The padded array of shape (n_images, new_size, new_size).
    """
    n_images, original_height, original_width = input_array.shape
    
    if new_size <= original_height or new_size <= original_width:
        raise ValueError("New size must be greater than the original dimensions.")
    
    # Calculate padding sizes
    padding_height = (new_size - original_height) // 2
    padding_width = (new_size - original_width) // 2
    
    # Create padded array
    padded_array = np.full((n_images, new_size, new_size), 1, dtype=input_array.dtype)
    
    # Make the background the mode colour
    from statistics import mode
    modes = [mode(i.flatten()) for i in input_array]
    padded_array = np.array([i*j for i, j in zip(padded_array, modes)])
    
    # Copy original images into the center of the new array
    for i in range(n_images):
        padded_array[i,
                     padding_height:padding_height + original_height,
                     padding_width:padding_width + original_width] = input_array[i]
    
    return padded_array
    