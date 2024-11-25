from keras.layers import Input, Conv2D, Dropout, AveragePooling2D, Conv2DTranspose, concatenate, Flatten, Dense, Reshape, Cropping2D
from tools import encoder_sub_block, encoder_mid_block, crop_and_merge
import keras.backend as K 
import numpy as np
#

def unet_model_1(IMG_SHAPE, RESULT_SHAPE):
    """
    Very roughly taken from BA project and slightly modified to fit larger data
    """
    config = {'activation': 'relu', 'conca': 'twice', 'decov': 'Conv2DTranspose', 'drop': 0.0655936065918576,
                'drop_mid': 0.21194937479704087, 'f1': 0, 'f2': 3, 'f3': 3, 'k1': 10, 'k2': 25, 'k3': 30,
                'ker_init': 'glorot_normal', 'pool_type': 'average'}
        
    input_net = Input(IMG_SHAPE)
    print("input_net shape:", input_net.shape)
    f_list = [(3, 3, 3, 3, 3, 3), (5, 5, 2, 2, 3, 3), (5, 5, 3, 3, 2, 3), (3, 3, 2, 2, 4, 3),
            (2, 4, 2, 4, 4, 2), (2, 2, 4, 3, 3, 3)]
    
    # set parameters
    f = f_list[config["f1"]]  # select which tuple from f_list
    k1 = config["k1"]
    activation = config["activation"]
    ker_init = config["ker_init"]
    drop = config["drop"]
    k2 = config["k2"]
    k3 = config["k3"]
    drop_mid = config["drop_mid"]
    f2 = config["f2"]
    f3 = config["f3"]
    
    # The encoder
    print("Encoder Shape")
    enc1 = Conv2D(k1, (f[0], f[0]), activation=activation, kernel_initializer=ker_init)(input_net)
    print("\n\nenc1 shape:", enc1.shape)
    d1 = Dropout(drop)(enc1)
    print("d1 shape:", d1.shape)
    enc2 = Conv2D(k1, (f[1], f[1]), activation=activation, kernel_initializer=ker_init)(d1)
    print("enc2 shape:", enc2.shape)
    enc3 = AveragePooling2D((2, 2))(enc2)
    print("enc3 shape:", enc3.shape)
    enc4 = Conv2D(k2, (f[2], f[2]), activation=activation, kernel_initializer=ker_init)(enc3)
    print("enc4 shape:", enc4.shape)
    d2 = Dropout(drop)(enc4)
    print("d2 shape:", d2.shape)
    enc5 = Conv2D(k2, (f[3], f[3]), activation=activation, kernel_initializer=ker_init)(d2)
    print("enc5 shape:", enc5.shape)
    enc6 = AveragePooling2D((2, 2))(enc5)
    print("enc6 shape:", enc6.shape)
    enc7 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(enc6)
    print("enc7 shape:", enc7.shape)
    d_mid = Dropout(drop_mid)(enc7)
    print("dmid shape:", d_mid.shape)
    enc8 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(d_mid)
    print("enc8 shape:", enc8.shape)
    
    print("\n")
    # input("\n [PAUSE] look at encoder shapes")
    
    # The decoder
    dec0 = Conv2DTranspose(k2, (2, 2), strides=(2, 2))(enc8)

    try:
        merge1 = concatenate([dec0, enc5])
    except ValueError:  # crop tensor if shapes do not match
        merge1 = crop_and_merge(dec0, enc5)

    dec1 = Conv2D(k2, (f[4], f[4]), activation=activation, kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(k2, (f[5], f[5]), activation=activation, kernel_initializer=ker_init, padding='same')(dec1)
    dec3 = Conv2DTranspose(k1, (2, 2), strides=(2, 2))(dec2)

    try:
        merge2 = concatenate([dec3, enc4])
    except ValueError:  # crop tensor if shapes do not match
        merge2 = crop_and_merge(dec3, enc4)



    dec4 = Conv2D(k1, (f2, f2), activation=activation, kernel_initializer=ker_init)(merge2)
    print("dec4 shape:", dec4.shape)
    dec5 = Conv2D(k1, (f3, f3), activation=activation, kernel_initializer=ker_init)(dec4)
    print("dec5 shape:", dec5.shape)
    # dec6 = Conv2DTranspose(k1, (3, 3), strides=(3, 3))(dec5)
    dec6 = dec5
    print("dec6 shape:", dec6.shape)
    dec7 = Flatten()(dec6)
    print("dec7 shape:", dec7.shape)
    dec8 = Dense(np.prod(RESULT_SHAPE))(dec7)
    print("dec8 shape:", dec8.shape)
    output_net = Reshape(RESULT_SHAPE)(dec8)  # 64, 64, 1
    print("output_net")
    return input_net, output_net

def unet_model_2(IMG_SHAPE, RESULT_SHAPE):
    """
    unet optimised by BA students and collated into a single object with no if statements
    """    
    
    config = {'activation': 'relu', 'conca': 'twice', 'decov': 'Conv2DTranspose', 'drop': 0.0655936065918576,
              'drop_mid': 0.21194937479704087, 'f1': 0, 'f2': 3, 'f3': 3, 'k1': 18, 'k2': 150, 'k3': 150,
              'ker_init': 'glorot_normal', 'pool_type': 'average'}
    
    input_net = Input(IMG_SHAPE)
    print("input_net shape:", input_net.shape)
    f_list = [(3, 3, 3, 3, 3, 3), (5, 5, 2, 2, 3, 3), (5, 5, 3, 3, 2, 3), (3, 3, 2, 2, 4, 3),
               (2, 4, 2, 4, 4, 2), (2, 2, 4, 3, 3, 3)]
    
    # set parameters
    f = f_list[config["f1"]]  # select which tuple from f_list
    k1 = config["k1"]
    activation = config["activation"]
    ker_init = config["ker_init"]
    drop = config["drop"]
    pool_type = config["pool_type"]
    k2 = config["k2"]
    k3 = config["k3"]
    drop_mid = config["drop_mid"]
    conca = config["conca"]
    decov = config["decov"]
    f2 = config["f2"]
    f3 = config["f3"]
    pool_size_1 = 4
    pool_size_2 = 2
    
    # The encoder
    print("Encoder Shape")
    enc1 = Conv2D(k1, (f[0], f[0]), activation=activation, kernel_initializer=ker_init)(input_net)
    print("\n\nenc1 shape:", enc1.shape)
    d1 = Dropout(drop)(enc1)
    print("d1 shape:", d1.shape)
    enc2 = Conv2D(k1, (f[1], f[1]), activation=activation, kernel_initializer=ker_init)(d1)
    print("enc2 shape:", enc2.shape)
    enc3 = AveragePooling2D((pool_size_1, pool_size_1))(enc2)
    print("enc3 shape:", enc3.shape)
    enc4 = Conv2D(k2, (f[2], f[2]), activation=activation, kernel_initializer=ker_init)(enc3)
    print("enc4 shape:", enc4.shape)
    d2 = Dropout(drop)(enc4)
    print("d2 shape:", d2.shape)
    enc5 = Conv2D(k2, (f[3], f[3]), activation=activation, kernel_initializer=ker_init)(d2)
    print("enc5 shape:", enc5.shape)
    enc6 = AveragePooling2D((pool_size_2, pool_size_2))(enc5)
    print("enc6 shape:", enc6.shape)
    enc7 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(enc6)
    print("enc7 shape:", enc7.shape)
    d_mid = Dropout(drop_mid)(enc7)
    print("dmid shape:", d_mid.shape)
    enc8 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(d_mid)
    print("enc8 shape:", enc8.shape)
    
    print("\n")
    # input("\n [PAUSE] look at encoder shapes")
    
    # The decoder
    dec0 = Conv2DTranspose(k2, (2, 2), strides=(2, 2))(enc8)

    try:
        merge1 = concatenate([dec0, enc5])
    except ValueError:  # crop tensor if shapes do not match
        shape_de = K.int_shape(dec0)[1]
        shape_en = K.int_shape(enc5)[1]
        diff = shape_de - shape_en

        if diff > 0:
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            dec0 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(dec0)

        elif diff < 0:
            diff = -int(1 * diff)
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            enc5 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(enc5)

        merge1 = concatenate([dec0, enc5])

    dec1 = Conv2D(k2, (f[4], f[4]), activation=activation, kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(k2, (f[5], f[5]), activation=activation, kernel_initializer=ker_init, padding='same')(dec1)
    dec3 = Conv2DTranspose(k1, (2, 2), strides=(2, 2))(dec2)

    try:
        merge2 = concatenate([dec3, enc4])
        print("dec and enc4 shapes", dec3.shape, enc4.shape)
    except:  # crop tensor if shapes do not match
        shape_de = K.int_shape(dec3)[1]
        shape_en = K.int_shape(enc4)[1]
        diff = shape_de - shape_en
        print("dec and enc4 shapes", dec3.shape, enc4.shape)
        if diff > 0:
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            dec3 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(dec3)
        elif diff < 0:
            diff = -int(1 * diff)
            diff_bottom = int(round(diff / 2))
            diff_top = int(diff - diff_bottom)
            enc4 = Cropping2D(cropping=((diff_top, diff_bottom), (diff_top, diff_bottom)))(enc4)

        merge2 = concatenate([dec3, enc4])



    dec4 = Conv2D(k1, (f2, f2), activation=activation, kernel_initializer=ker_init)(merge2)
    print("dec4 shape:", dec4.shape)
    dec5 = Conv2D(k1, (f3, f3), activation=activation, kernel_initializer=ker_init)(dec4)
    print("dec5 shape:", dec5.shape)
    # dec6 = Conv2DTranspose(k1, (3, 3), strides=(3, 3))(dec5)
    dec6 = dec5
    print("dec6 shape:", dec6.shape)
    dec7 = Flatten()(dec6)
    print("dec7 shape:", dec7.shape)
    dec8 = Dense(np.prod(RESULT_SHAPE))(dec7)
    print("dec8 shape:", dec8.shape)
    output_net = Reshape(RESULT_SHAPE)(dec8)  # 64, 64, 1
    print("output_net")
    return input_net, output_net

def unet_model_3(IMG_SHAPE, RESULT_SHAPE):
    """
    keeping the same activation, upscaling, dropouts, initialisation and pool-typeing
    greater considerations to new dimensions
    """
        
    input_net = Input(IMG_SHAPE) # make inputs 138,138
    
    # set parameters
    ker_init = 'glorot_normal'
    drop = 0.0655936065918576
    drop_mid = 0.21194937479704087
    

    # ENCODER
    # d = 138,138,1 -> 134,134,40 -> 132,132,80 -> 44,44,80
    enc3 = encoder_sub_block(input_net, NumFilter1=40, NumFilter2=80, KernelSize1=(5,5), 
                      KernelSize2=(3,3), DropoutRate=drop, DonwsampleSize=(3,3), ker_init=ker_init)

    # -> 40,40,120 -> 36,36,120 -> 12,12,120
    enc4, enc5, enc6 = encoder_sub_block(enc3, 120, 120, (5,5),(5,5), drop, (3,3), ker_init, all_encoders=True)

    # -> 12,12,150 -> 12,12,150
    enc8 = encoder_mid_block(enc6, NumFilter1=150, NumFilter2=150, DropoutMid=drop_mid, ker_init=ker_init)


    ## DECODER
    # -> 36,36,120 -> 36,36,240
    dec0 = Conv2DTranspose(120, (3, 3), strides=(3, 3))(enc8)
    merge1 = concatenate([dec0, enc5])
    
    # -> 32,32,120 -> 28,28,110 -> 24,24,100 -> 22,22,90-> 44,44,80
    dec1 = Conv2D(120, (5, 5), activation='relu', kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(110, (5, 5), activation='relu', kernel_initializer=ker_init)(dec1)
    dec3 = Conv2D(100, (5, 5), activation='relu', kernel_initializer=ker_init)(dec2)
    dec4 = Conv2D(90, (3,3), activation='relu', kernel_initializer=ker_init)(dec3)
    dec5 = Conv2DTranspose(80, (2, 2), strides=(2, 2))(dec4)
    
    # -> 44,44,160 -> 42,42,40 -> 40,40,3
    merge2 = concatenate([dec5, enc3])
    dec6 = Conv2D(40, (3, 3), activation='relu', kernel_initializer=ker_init)(merge2)
    dec7 = Conv2D(4, (3, 3), activation='relu', kernel_initializer=ker_init)(dec6)
    
    # -> 4800
    dec8 = Flatten()(dec7)
    # 4096 (=64^2)
    dec9 = Dense(np.prod(RESULT_SHAPE))(dec8)
    output_net = Reshape(RESULT_SHAPE)(dec9)

    return input_net, output_net


def unet_model_4(IMG_SHAPE, RESULT_SHAPE):
    """
    Same as model 3 but with reduced number of kernels to help reduce size
    """
        
    input_net = Input(IMG_SHAPE) # make inputs 138,138
    
    # set parameters
    ker_init = 'glorot_normal'
    drop = 0.0655936065918576
    drop_mid = 0.21194937479704087
    
    F_list = [20, 40, 60, 75, 4]    
    
    # ENCODER
    enc3 = encoder_sub_block(input_net, NumFilter1=F_list[0], NumFilter2=F_list[1], KernelSize1=(5,5), 
                      KernelSize2=(3,3), DropoutRate=drop, DonwsampleSize=(3,3), ker_init=ker_init)

    enc4, enc5, enc6 = encoder_sub_block(enc3, F_list[2], F_list[2], (5,5),(5,5), drop, (3,3), ker_init, all_encoders=True)
    enc8 = encoder_mid_block(enc6, NumFilter1=F_list[3], NumFilter2=F_list[3], DropoutMid=drop_mid, ker_init=ker_init)


    ## DECODER
    dec0 = Conv2DTranspose(F_list[2], (3, 3), strides=(3, 3))(enc8)
    merge1 = concatenate([dec0, enc5])
    
    dec1 = Conv2D(F_list[2], (5, 5), activation='relu', kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(55, (5, 5), activation='relu', kernel_initializer=ker_init)(dec1)
    dec3 = Conv2D(50, (5, 5), activation='relu', kernel_initializer=ker_init)(dec2)
    dec4 = Conv2D(45, (3,3), activation='relu', kernel_initializer=ker_init)(dec3)
    dec5 = Conv2DTranspose(F_list[1], (2, 2), strides=(2, 2))(dec4)
    
    merge2 = concatenate([dec5, enc3])
    dec6 = Conv2D(F_list[1], (3, 3), activation='relu', kernel_initializer=ker_init)(merge2)
    dec7 = Conv2D(F_list[4], (3, 3), activation='relu', kernel_initializer=ker_init)(dec6)
    
    dec8 = Flatten()(dec7)
    dec9 = Dense(np.prod(RESULT_SHAPE))(dec8)
    output_net = Reshape(RESULT_SHAPE)(dec9)

    return input_net, output_net

def unet_model_5(IMG_SHAPE, RESULT_SHAPE):
    """
    Same as model 3 but with reduced number of kernels to help reduce size
    """
        
    input_net = Input(IMG_SHAPE) # make inputs 138,138
    
    # set parameters
    ker_init = 'glorot_normal'
    drop = 0.0655936065918576
    drop_mid = 0.21194937479704087
    
    F_list = [40, 80, 120, 150, 3]    
    
    # ENCODER
    enc3 = encoder_sub_block(input_net, NumFilter1=F_list[0], NumFilter2=F_list[1], KernelSize1=(5,5), 
                      KernelSize2=(3,3), DropoutRate=drop, DonwsampleSize=(3,3), ker_init=ker_init)

    enc4, enc5, enc6 = encoder_sub_block(enc3, F_list[2], F_list[2], (5,5),(5,5), drop, (3,3), ker_init, all_encoders=True)
    enc8 = encoder_mid_block(enc6, NumFilter1=F_list[3], NumFilter2=F_list[3], DropoutMid=drop_mid, ker_init=ker_init)


    ## DECODER
    dec0 = Conv2DTranspose(F_list[2], (3, 3), strides=(3, 3))(enc8)
    merge1 = concatenate([dec0, enc5])
    
    dec1 = Conv2D(F_list[2], (5, 5), activation='relu', kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(110, (5, 5), activation='relu', kernel_initializer=ker_init)(dec1)
    dec3 = Conv2D(100, (5, 5), activation='relu', kernel_initializer=ker_init)(dec2)
    dec4 = Conv2D(90, (3,3), activation='relu', kernel_initializer=ker_init)(dec3)
    dec5 = Conv2DTranspose(F_list[1], (2, 2), strides=(2, 2))(dec4)
    
    merge2 = concatenate([dec5, enc3])
    dec6 = Conv2D(F_list[1], (3, 3), activation='relu', kernel_initializer=ker_init)(merge2)
    dec7 = Conv2D(F_list[4], (3, 3), activation='relu', kernel_initializer=ker_init)(dec6)
    
    dec8 = Flatten()(dec7)
    dec9 = Dense(np.prod(RESULT_SHAPE))(dec8)
    output_net = Reshape(RESULT_SHAPE)(dec9)

    return input_net, output_net

def unet_model_6(IMG_SHAPE, RESULT_SHAPE):
    """
    Same as model 3 but with reduced number of kernels to help reduce size
    """
        
    input_net = Input(IMG_SHAPE) # make inputs 138,138
    
    # set parameters
    ker_init = 'glorot_normal'
    drop = 0.0655936065918576
    drop_mid = 0.21194937479704087
    
    F_list = [40, 80, 120, 150, 5]    
    
    # ENCODER
    enc3 = encoder_sub_block(input_net, NumFilter1=F_list[0], NumFilter2=F_list[1], KernelSize1=(5,5), 
                      KernelSize2=(3,3), DropoutRate=drop, DonwsampleSize=(3,3), ker_init=ker_init)

    enc4, enc5, enc6 = encoder_sub_block(enc3, F_list[2], F_list[2], (5,5),(5,5), drop, (3,3), ker_init, all_encoders=True)
    enc8 = encoder_mid_block(enc6, NumFilter1=F_list[3], NumFilter2=F_list[3], DropoutMid=drop_mid, ker_init=ker_init)


    ## DECODER
    dec0 = Conv2DTranspose(F_list[2], (3, 3), strides=(3, 3))(enc8)
    merge1 = concatenate([dec0, enc5])
    
    dec1 = Conv2D(F_list[2], (5, 5), activation='relu', kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(110, (5, 5), activation='relu', kernel_initializer=ker_init)(dec1)
    dec3 = Conv2D(100, (5, 5), activation='relu', kernel_initializer=ker_init)(dec2)
    dec4 = Conv2D(90, (3,3), activation='relu', kernel_initializer=ker_init)(dec3)
    dec5 = Conv2DTranspose(F_list[1], (2, 2), strides=(2, 2))(dec4)
    
    merge2 = concatenate([dec5, enc3])
    dec6 = Conv2D(F_list[1], (3, 3), activation='relu', kernel_initializer=ker_init)(merge2)
    dec7 = Conv2D(F_list[4], (3, 3), activation='relu', kernel_initializer=ker_init)(dec6)
    
    dec8 = Flatten()(dec7)
    dec9 = Dense(np.prod(RESULT_SHAPE))(dec8)
    output_net = Reshape(RESULT_SHAPE)(dec9)

    return input_net, output_net

def unet_model_7(IMG_SHAPE, RESULT_SHAPE):
    """
    Same as model 3 but with reduced number of kernels to help reduce size
    """
        
    input_net = Input(IMG_SHAPE) # make inputs 138,138
    
    # set parameters
    ker_init = 'glorot_normal'
    drop = 0.0655936065918576
    drop_mid = 0.21194937479704087
    
    F_list = [80, 110, 150, 200, 4]      
    
    # ENCODER
    enc3 = encoder_sub_block(input_net, NumFilter1=F_list[0], NumFilter2=F_list[1], KernelSize1=(5,5), 
                      KernelSize2=(3,3), DropoutRate=drop, DonwsampleSize=(3,3), ker_init=ker_init)

    enc4, enc5, enc6 = encoder_sub_block(enc3, F_list[2], F_list[2], (5,5),(5,5), drop, (3,3), ker_init, all_encoders=True)
    enc8 = encoder_mid_block(enc6, NumFilter1=F_list[3], NumFilter2=F_list[3], DropoutMid=drop_mid, ker_init=ker_init)


    ## DECODER
    dec0 = Conv2DTranspose(F_list[2], (3, 3), strides=(3, 3))(enc8)
    merge1 = concatenate([dec0, enc5])
    
    dec1 = Conv2D(F_list[2], (5, 5), activation='relu', kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(140, (5, 5), activation='relu', kernel_initializer=ker_init)(dec1)
    dec3 = Conv2D(130, (5, 5), activation='relu', kernel_initializer=ker_init)(dec2)
    dec4 = Conv2D(120, (3,3), activation='relu', kernel_initializer=ker_init)(dec3)
    dec5 = Conv2DTranspose(F_list[1], (2, 2), strides=(2, 2))(dec4)
    
    merge2 = concatenate([dec5, enc3])
    dec6 = Conv2D(F_list[1], (3, 3), activation='relu', kernel_initializer=ker_init)(merge2)
    dec7 = Conv2D(F_list[4], (3, 3), activation='relu', kernel_initializer=ker_init)(dec6)
    
    dec8 = Flatten()(dec7)
    dec9 = Dense(np.prod(RESULT_SHAPE))(dec8)
    output_net = Reshape(RESULT_SHAPE)(dec9)

    return input_net, output_net


def unet_model_small(IMG_SHAPE, RESULT_SHAPE):
    """
    Same as model 3 but with reduced number of kernels to help reduce size
    """
        
    input_net = Input(IMG_SHAPE) # make inputs 138,138
    
    # set parameters
    ker_init = 'glorot_normal'
    drop = 0.0655936065918576
    drop_mid = 0.21194937479704087
    
    F_list = [20, 40, 60, 75, 3]    
    
    # ENCODER
    enc3 = encoder_sub_block(input_net, NumFilter1=F_list[0], NumFilter2=F_list[1], KernelSize1=(5,5), 
                      KernelSize2=(3,3), DropoutRate=drop, DonwsampleSize=(3,3), ker_init=ker_init)

    enc4, enc5, enc6 = encoder_sub_block(enc3, F_list[2], F_list[2], (5,5),(5,5), drop, (3,3), ker_init, all_encoders=True)
    enc8 = encoder_mid_block(enc6, NumFilter1=F_list[3], NumFilter2=F_list[3], DropoutMid=drop_mid, ker_init=ker_init)


    ## DECODER
    dec0 = Conv2DTranspose(F_list[2], (3, 3), strides=(3, 3))(enc8)
    merge1 = concatenate([dec0, enc5])
    
    dec1 = Conv2D(F_list[2], (5, 5), activation='relu', kernel_initializer=ker_init)(merge1)
    dec2 = Conv2D(55, (5, 5), activation='relu', kernel_initializer=ker_init)(dec1)
    dec3 = Conv2D(50, (5, 5), activation='relu', kernel_initializer=ker_init)(dec2)
    dec4 = Conv2D(45, (3,3), activation='relu', kernel_initializer=ker_init)(dec3)
    dec5 = Conv2DTranspose(F_list[1], (2, 2), strides=(2, 2))(dec4)
    
    merge2 = concatenate([dec5, enc3])
    dec6 = Conv2D(F_list[1], (3, 3), activation='relu', kernel_initializer=ker_init)(merge2)
    dec7 = Conv2D(F_list[4], (3, 3), activation='relu', kernel_initializer=ker_init)(dec6)
    
    dec8 = Flatten()(dec7)
    dec9 = Dense(np.prod(RESULT_SHAPE))(dec8)
    output_net = Reshape(RESULT_SHAPE)(dec9)

    return input_net, output_net