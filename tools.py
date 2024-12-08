from keras.layers import Conv2D, Dropout, AveragePooling2D, concatenate, Cropping2D # type: ignore
import keras.backend as K # type: ignore
import numpy as np # type: ignore 
from tensorflow.keras.callbacks import Callback # type: ignore

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
    ----------
    Parameters
    ----------
        input_array; ndarray 
            The input array of shape (n_images, height, width).
        
        new_size: int 
            The desired new width and height of the padded images.
        
        padding_value: int or float
            The value to use for padding. Default is 0.
        
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
    padded_array = np.full((n_images, new_size, new_size), padding_value, dtype=input_array.dtype)
    # padded_array = np.full((n_images, new_size, new_size), 1, dtype=input_array.dtype)

    # from statistics import mode
    # modes = [mode(i.flatten()) for i in input_array]
    # padded_array = np.array([i*j for i, j in zip(modes, padded_array)])

    # Copy original images into the center of the new array
    for i in range(n_images):
        padded_array[i,
                     padding_height:padding_height + original_height,
                     padding_width:padding_width + original_width] = input_array[i]
    
    return padded_array

def resample_image_by_scale(image: np.ndarray, new_width: int) -> np.ndarray:
    """
    Takes a square image and resamples it to another square image of different 
    dimensions
    Principle behined it is finds the lowest common multiple of the new and old 
    dimensions, then scales it up using np.repeat to be the size of the lcm, 
    then takes an average over an intiger number of squares to downsample to 
    new dimensions
    ----------
    Parameters
    ----------
    image : np.ndarray
        Square (n,n) array that will be resampled
    new_width : int
        width of new resampled image

    Returns
    -------
    new_image : np.ndarray
        new resampled image dimesnsions (new_width, new_width)
        
    None if not a square
    """
    
    if image.shape[0] != image.shape[1]:
        print("Not a square image")
        return None
        
    n = image.shape[0]
    m = new_width
    
    from math import lcm
    
    lcm_mn = int(lcm(n,m))
    
    multiply_scale = int(lcm_mn/n)
    reduction_scale = int(lcm_mn/m)
    
    # scales it up
    new_image = np.repeat(image, multiply_scale, axis=1)
    new_image = np.repeat(new_image, multiply_scale, axis=0)
    
    # scales it down using the average
    new_image = np.array([np.average(new_image[i:i+reduction_scale,j:j+reduction_scale].flatten()) for i in range(0, lcm_mn, reduction_scale) for j in range(0, lcm_mn, reduction_scale)]).reshape((m,m))
    
    return new_image

def resample_imporved(images: np.ndarray, new_width: int) -> np.ndarray:
    """
    Takes a square image and resamples it to another square image of different 
    dimensions
    Principle behined it is finds the lowest common multiple of the new and old 
    dimensions, then scales it up using np.repeat to be the size of the lcm, 
    then takes an average over an intiger number of squares to downsample to 
    new dimensions

    improved time and efficiency of resample_image_by_scale() 
    ----------
    Parameters
    ----------
    image : np.ndarray
        Square (n,n) array that will be resampled
    new_width : int
        width of new resampled image

    -------
    Returns
    -------
    new_image : np.ndarray
        new resampled image dimesnsions (new_width, new_width)
        
    None if not a square
    """
    if images.shape[1] != images.shape[2]:
        print("Not a square image")
        return None
    
    total = len(images)
    n = images.shape[1]
    m = new_width

    if m == n:
        return images
    
    from math import lcm
    
    lcm_mn = int(lcm(n,m))
    
    multiply_scale = int(lcm_mn/n)
    reduction_scale = int(lcm_mn/m)
    
    new_images = np.repeat(np.repeat(images, multiply_scale, axis=1), multiply_scale, axis=2)
    if reduction_scale == 1:
        return new_images

    
    new_images = new_images.reshape(total, m*reduction_scale, m, reduction_scale).mean(axis=-1)
    rot_image = np.rot90(new_images, 1, (1,2))
    new_images = rot_image.reshape(total, m, m, reduction_scale).mean(axis=-1).reshape(total,m,m)
    un_rot_image = np.rot90(new_images, -1, (1,2))
    
    return un_rot_image

def resample_image_sp(images, new_size):
    """
    uses the scipy zoom function to rescale images using a 5th order interpolation
    works on an entire array of images
    ----------
    Parameters
    ----------
    image : np.ndarray
        Square (n,n) array that will be resampled
    new_width : int
        width of new resampled image

    -------
    Returns
    -------
    new_image : np.ndarray
        new resampled image dimesnsions (new_width, new_width)
    """
    zoom_scale = new_size/images.shape[1]
    from scipy.ndimage import zoom # type: ignore
    return np.array([zoom(im, zoom_scale, order=5) for im in images])

def spectrum_retrival(paths):
    import datasets as ds # type: ignore
    
    total = ds.load(paths[0]).raw
    for i in paths[1:]:
        temp = ds.load(i).raw
        total = np.concatenate((total, temp), axis=0)
    n = len(total)
    total = total[:,10:74,65:193].reshape(n,64,64,2).mean(axis=-1).reshape(n,64,64)
        
    return total

def multiplied(data, mult):
    """
    multiplys each value in any dimentional array by mult
    """
    mult_matrix = np.full(data.shape, mult)
    return data*mult_matrix
    # another method that when tested was marginaly slower
    # return np.array(data.flatten() * mult).reshape(data.shape)

def log_scale_data(data, undo=False):
    """
    transforms data values b:y y = log(x+1) 
    undo will do inverse transform: x = e^(y) - 1
    """
    if undo:
        transformed_data = np.array(np.e**data.flatten() - 1)
    else:
        transformed_data = np.array(np.log(data.flatten() + 1)).reshape(data.shape)
    return transformed_data.reshape(data.shape)

def norm(data, reduction_factor=None):
    if not reduction_factor:
        flat_data = data.flatten()
        reduction_factor = max(flat_data)
        return (flat_data/reduction_factor).reshape(data.shape), reduction_factor
    else:
        flat_data = data.flatten()
        return (flat_data/reduction_factor).reshape(data.shape)



class TrendBasedEarlyStopping(Callback):
    def __init__(self, monitor_train='loss', monitor_val='val_loss', patience=5):
        super(TrendBasedEarlyStopping, self).__init__()
        self.monitor_train = monitor_train
        self.monitor_val = monitor_val
        self.patience = patience
        self.diff_history = []  # To store the differences for the last `patience` epochs
        self.best_weights = None  # To store the best weights

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get(self.monitor_train)
        val_loss = logs.get(self.monitor_val)
        if train_loss is None or val_loss is None:
            return  # Skip if the metrics are not available

        # Calculate the difference between val_loss and train_loss
        diff = val_loss - train_loss
        self.diff_history.append(diff)

        # Keep only the last `patience` differences
        if len(self.diff_history) > self.patience:
            self.diff_history.pop(0)

        # Check if the differences are strictly increasing
        if len(self.diff_history) == self.patience and all(
            self.diff_history[i] < self.diff_history[i + 1] for i in range(len(self.diff_history) - 1)
        ):
            print(f"Epoch {epoch + 1}: Overfitting trend detected (increasing val_loss - train_loss).")
            print("Stopping training and reverting to the best weights from earlier epochs.")
            self.model.stop_training = True
            if self.best_weights:
                self.model.set_weights(self.best_weights)
        
        # Save the best weights based on val_loss
        if len(self.diff_history) == 1 or val_loss < min(self.diff_history):
            self.best_weights = self.model.get_weights()

class DivergenceEarlyStopping(Callback):
    def __init__(self, monitor_train='loss', monitor_val='val_loss', patience=7):
        super(DivergenceEarlyStopping, self).__init__()
        self.patience = patience
        self.monitor_train = monitor_train
        self.monitor_val = monitor_val
        self.best_weights = None
        self.divergence_history = []
        self.best_val_loss = np.inf
        self.initial_wait = 30

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get(self.monitor_train)
        val_loss = logs.get(self.monitor_val)
        if train_loss is None or val_loss is None:
            return  # Skip if the metrics are not available

        diff = abs(train_loss - val_loss)

        if epoch < self.initial_wait:
            self.best_weights = self.model.get_weights()
            return
        
        self.divergence_history.append(diff)

        if val_loss < self.best_val_loss and all(diff < i for i in self.divergence_history):
            self.best_val_loss = val_loss
            self.best_weights = self.model.get_weights()
            self.divergence_history = []
            return

        if len(self.divergence_history) == self.patience:
            print("\nTraining halted early:")
            print("\tEither train and validation values are diverging or validation loss is not decreasing.\n")
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


        


        


