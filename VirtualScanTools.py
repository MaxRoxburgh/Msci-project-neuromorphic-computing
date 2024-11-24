def take_scan_virtual(data, model_path, mult, power, padding_value=0, invert=False):
    """
    This script takes in data like in the physical system, but loads a keras model and does the prediction    
    ----------
    Parameters
    ----------
    data: ndarray
        This is the raw image data that is to be fed into the system images stacked on axis=0 and must be square.
        
    model_path: str
        Path directory to where the model is stored.
        
    mult: int 
        Same as in take_scan() from experiment
            Images have a multiplication factor that scales up the images to fit within a total boarder of 532
            This code finds the length for experiment then scales it for the model
            
    power: int
        power of the laser in the experiment.
        
    padding_value : int, optional
        If images do not fill the full space, padding is added usualy the modal pixel strength in the data. 
        The default is 0.
        
    invert: bool, optional
        If the pixel data is to be inverted. 
        The default is False.
        
    Returns
    -------
    ndarray
        predicted spectra from experiment.

    """
    
    from tools import add_padding_to_images, resample_imporved
    from tensorflow.keras.models import load_model
    import gc
    
    scan_pic_width = data.shape[1]*mult
    new_size = int(scan_pic_width/4)
    
    if invert:
        rescaled_data = 255 - resample_imporved(data, new_size)
    else:
        rescaled_data = resample_imporved(data, new_size)
    
    rescaled_data *= power
    
    if new_size != 133:
        rescaled_data = add_padding_to_images(rescaled_data, 133, padding_value*power)
    
    full_padded_data = add_padding_to_images(rescaled_data, 138, 0)
    
    del rescaled_data
    gc.collect()
    
    model = load_model(model_path)
        
    return model.predict(full_padded_data)


    
    
    
    
    