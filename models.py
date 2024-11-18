from keras.layers import *
from tools import enocoder_sub_block

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
    pool_type = config["pool_type"]
    k2 = config["k2"]
    k3 = config["k3"]
    drop_mid = config["drop_mid"]
    conca = config["conca"]
    decov = config["decov"]
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
    except:  # crop tensor if shapes do not match
        shape_de = K.int_shape(dec3)[1]
        shape_en = K.int_shape(enc4)[1]
        diff = shape_de - shape_en

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

def unet_model_2(IMG_SHAPE, RESULT_SHAPE):
    """
    keeping the same activation, upscaling, dropouts, initialisation and pool-typeing
    greater considerations to new dimensions
    """
    config = {'activation': 'relu', 'conca': 'twice', 'decov': 'Conv2DTranspose', 'drop': 0.0655936065918576,
                'drop_mid': 0.21194937479704087, 'f1': 0, 'f2': 3, 'f3': 3, 'k1': 10, 'k2': 25, 'k3': 30,
                'ker_init': 'glorot_normal', 'pool_type': 'average'}
        
    input_net = Input(IMG_SHAPE) # make inputs 138,138
    # print("input_net shape:", input_net.shape)
    
    # set parameters
    activation = 'relu'
    ker_init = 'glorot_normal'
    drop = 0.0655936065918576
    pool_type = 'average'
    drop_mid = 0.21194937479704087
    decov = 'Conv2DTranspose'


    # print("Encoder Shape")

    # picked 40 as there were ~40 features that the netSALT simulation found 
    enc1 = Conv2D(40, (5, 5), activation=activation, kernel_initializer=ker_init)(input_net) # d = 134,134,40
    # print("\n\nenc1 shape:", enc1.shape)
    d1 = Dropout(drop)(enc1)
    enc2 = Conv2D(80, (3, 3), activation=activation, kernel_initializer=ker_init)(d1) # d = 132,132,80
    # print("enc2 shape:", enc2.shape)
    enc3 = AveragePooling2D((3, 3))(enc2) # d = 44,44,80   

    #################################################################
    # above block encoded into a single function
    # enc3 = encoder_sub_block(input_net, NumFilter1=40, NumFilter2=80, KernelSize1=(5,5), 
    #                   KernelSize2=(3,3), DropoutRate=drop, DonwsampleSize=(3,3), ker_init=ker_init)
    #################################################################

    enc6 = enocoder_sub_block(enc3, 120, 120, (5,5),(5,5), drop, (3,3), ker_init)
    # d = 40,40,120 -> d = 36,36,120 -> d = 12,12,120



    # enc7 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(enc6)
    # # print("enc7 shape:", enc7.shape)
    # d_mid = Dropout(drop_mid)(enc7)
    # # print("dmid shape:", d_mid.shape)
    # enc8 = Conv2D(k3, (1, 1), activation=activation, kernel_initializer=ker_init)(d_mid)
    # # print("enc8 shape:", enc8.shape)


    pass

def unet_model_3(IMG_SHAPE, RESULT_SHAPE):
    pass

def unet_model_4(IMG_SHAPE, RESULT_SHAPE):
    pass

def unet_model_5(IMG_SHAPE, RESULT_SHAPE):
    pass