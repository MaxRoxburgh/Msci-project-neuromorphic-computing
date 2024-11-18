from keras.layers import *

def encoder_sub_block(input, NumFilter1, NumFilter2, KernelSize1, KernelSize2, DropoutRate, DonwsampleSize, ker_init='random_normal', activation='relu', debug=False):
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

    return enc3

def encoder_mid_block(input, NumFilter1, NumFilter2, DropoutMid, activation, ker_init, debug=False):
    """
    
    """
    enc1 = Conv2D(NumFilter1, (1, 1), activation=activation, kernel_initializer=ker_init)(enc6)
    # print("enc7 shape:", enc7.shape)
    d_mid = Dropout(DropoutMid)(enc1)
    # print("dmid shape:", d_mid.shape)
    enc2 = Conv2D(NumFilter2, (1, 1), activation=activation, kernel_initializer=ker_init)(d_mid)
    # print("enc8 shape:", enc8.shape)

    return enc2