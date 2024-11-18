from keras.layers import *
import keras.backend as K 

def encoder_sub_block(input, NumFilter1, NumFilter2, KernelSize1, KernelSize2, DropoutRate, DonwsampleSize, ker_init='random_normal', activation='relu', debug=False, all_encoders=False):
    """
    two convolutional layers seperated by a dropout layer, AveragePooling2D after
    PARAMETERS:
        input: ndarray 
            input_net

        NumFilter1: int 
            number of convolutional filters in the first layer

        NumFilter2: int
            number of convolutional filters in the second layer

        KernelSize1: (int, int)
            convolutional kernel size for first convolutional layer

        KernelSize2: (int, int)
            convolutional kernel size for second convolutional layer

        DropoutRate: float
            dropout between the first and second layer

        DonwsampleSize: (int, int)
            the kernel for the average pooling layer
        
        ker_init: str
            initialisation type of the 

        activation: str
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
    PARAMETERS
        input: ndarray 
            input_net

        NumFilter1: int 
            number of convolutional filters in the first layer

        NumFilter2: int
            number of convolutional filters in the second layer
        
        DropoutMid: float   
            dropout between the first and second layer
        
        ker_init: str
            initialisation type of the 

        activation: str
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
    