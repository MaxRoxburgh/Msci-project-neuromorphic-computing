"""
Plotting functions
"""

import os
import io
import sys
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import cv2
from sklearn import datasets
from skimage import color, filters, feature, measure


def losshistory(history, dir_folder, plot_show):
    """
    Plot and save loss history in both log and normal scale
    """

    fig, host = plt.subplots()
    ax2 = host.twinx()
    ax3 = host.twinx()

    print(history.keys())
    print()

    p1 = host.semilogy(history['loss'], label='train loss')
    p2 = host.semilogy(history['val_loss'], label='test loss')
    p3 = ax2.semilogy(history['accuracy'], color='C2', label='train accuracy')
    p4 = ax3.semilogy(history['val_accuracy'], color='C3', label='test accuracy')

    host.set_xlabel('# of epochs')
    host.set_ylabel('MSE loss')
    ax2.set_ylabel('Train accuracy')
    ax3.set_ylabel('Test accuracy')

    host.legend(handles=p1+p2+p3+p4, loc='center right')

    ax2.yaxis.label.set_color(p3[0].get_color())
    ax3.yaxis.label.set_color(p4[0].get_color())
    ax3.spines['right'].set_position(('outward', 60))

    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                bbox_inches='tight')
    png2 = Image.open(png1)
    png2.save(dir_folder + "/model_accuracy-loss_log.tiff")
    png1.close()
    if plot_show:  # show plot if True
        plt.show()

    fig, host = plt.subplots()
    ax2 = host.twinx()

    p1 = host.plot(history['loss'], label='train loss')
    p2 = host.plot(history['val_loss'], label='validation loss')
    p3 = ax2.plot(history['accuracy'], color='C2', label='train accuracy')
    p4 = ax2.plot(history['val_accuracy'], color='C3', label='validation accuracy')

    host.set_xlabel('# of epochs')
    host.set_ylabel('Train and validation MAE loss')
    ax2.set_ylabel('Train and validation accuracy')

    host.legend(handles=p1+p2+p3+p4, loc='center right')

    ax2.yaxis.label.set_color(p3[0].get_color())
    ax3.yaxis.label.set_color(p4[0].get_color())
    ax3.spines['right'].set_position(('outward', 60))

    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                bbox_inches='tight')
    png2 = Image.open(png1)
    png2.save(dir_folder + "/model_accuracy-loss.tiff")
    png1.close()
    if plot_show:  # show plot if True
        plt.show()

    print('Final train loss:', history['loss'][-1])
    print('Final test loss:', history['val_loss'][-1])
    print('Final train accuracy:', history['accuracy'][-1])
    print('Final test accuracy:', history['val_accuracy'][-1])


def visualize(img, train, encoder, decoder, model, use_unet):
    """
    Draw original, encoded and decoded images

    Adapted from:
    https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/
    """

    if use_unet:
        reco = model.predict(img[None], verbose=0)[0]
    else:
        # img[None] will have shape of (1, 32, 32, 3)
        # which is the same as the model input
        code = encoder.predict(img[None], verbose=0)[0]
        reco = decoder.predict(code[None], verbose=0)[0]

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title("Expected")
    plt.imshow(train)

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    plt.imshow(reco)


def predict_spec(img, reco):
    """
    Draw original mnist digit and predicted spectrum

    Adapted from:
    https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/
    """

    # plotting
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(reco)


# load digit image
def set_input_data(data_start, data_end):
    data_start, data_end = data_start - 1, data_end
    
    def rotate_input(input, rotate):
        if rotate == 0:
            return input
        if rotate == 90:
            return cv2.rotate(input, cv2.ROTATE_90_CLOCKWISE)
        if rotate == 180:
            return cv2.rotate(input, cv2.ROTATE_180)
        if rotate == 270:
            return cv2.rotate(input, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    input_data = {}

    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False, parser='auto', return_X_y=True)

    gray_images = list(
        mnist[0].reshape(
            -1, 28, 28).astype(np.uint8)[data_start:data_end])  # first 40k mnist images

    # binarisation of rotated images to either 0 or 255
    # image grey scale between 0 to 255, 127 is the mid-point
    threshold_gray = 127
    input_data["0"] = np.array([
        cv2.threshold(
            image, threshold_gray, 255, cv2.THRESH_BINARY)[1]
        for image in gray_images])
    input_data["90"] = np.array([
        cv2.threshold(
            rotate_input(image, 90), threshold_gray, 255,
            cv2.THRESH_BINARY)[1]
        for image in gray_images])
    input_data["180"] = np.array([
        cv2.threshold(
            rotate_input(image, 180), threshold_gray, 255,
            cv2.THRESH_BINARY)[1]
        for image in gray_images])
    input_data["270"] = np.array([
        cv2.threshold(
            rotate_input(image, 270), threshold_gray, 255,
            cv2.THRESH_BINARY)[1]
        for image in gray_images])

    print('MNIST rotation data produced')

    # binarisation of prewitt images to either 0 or 255
    # notation - x3n : left, x3p : right, y3n : top, y3p : bottom
    # image grey scale between -1 and 1, 0 is the mid point
    threshold_prewitt = 0
    input_data["x3n"] = np.array([
        cv2.threshold(
            filters.prewitt_v(image), threshold_prewitt, 255,
            cv2.THRESH_BINARY)[1]
        for image in gray_images])
    input_data["x3p"] = np.array([
        cv2.threshold(
            -filters.prewitt_v(image), threshold_prewitt, 255,
            cv2.THRESH_BINARY)[1]
        for image in gray_images])
    input_data["y3n"] = np.array([
        cv2.threshold(
            filters.prewitt_h(image), threshold_prewitt, 255,
            cv2.THRESH_BINARY)[1]
        for image in gray_images])
    input_data["y3p"] = np.array([
        cv2.threshold(
            -filters.prewitt_h(image), threshold_prewitt, 255,
            cv2.THRESH_BINARY)[1]
        for image in gray_images])
    # canny filter gives image in either 0 or 1
    # any number in between (0.1 used here) acts as the threshold of
    # binerisation
    input_data["canny"] = [
        feature.canny(image) * 1.0 for image in gray_images]
    input_data["canny"] = np.array([
        cv2.threshold(image, 0.1, 255, cv2.THRESH_BINARY)[1]
        for image in input_data["canny"]])

    print('MNIST edges and canny data produced')

    label = mnist[1][data_start:data_end]
    
    return input_data, label


def plot_digit(input_data, labels, data_range):
    keys = input_data.keys()
    keys_label = ['0 deg', '90 deg', '180 deg', '270 deg', 'Left edge', 'Right edge', 'Top edge', 'Bottom edge', 'Canny edge']
    for i, label in enumerate(labels):
        for j, key in enumerate(keys):
            digit = input_data[key]
            order = data_range[i]

            # plot
            plt.figure(figsize=(2, 2))
            plt.title(f"Digit {label} {keys_label[j]}")
            plt.imshow(digit[i], cmap='gray')
            # plt.colorbar()

            # directory path
            dir_path = os.path.dirname(os.path.realpath(__file__))
            sys.path.append(dir_path)

            # save
            save_name = f'No{order}_digit{label}_{key}.tiff'
            png1 = io.BytesIO()
            plt.savefig(png1, format="png", dpi=500, pad_inches=.1,
                        bbox_inches='tight')
            png2 = Image.open(png1)
            png2.save(dir_path + '/' + save_name)
            png1.close()


def replot_his():
    """
    Reload loss histroy dictionary and plot
    """

    # set the directory path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dir_path)

    # reload loss history and plot
    with open(dir_path + '/losshistory-dict', "rb") as file:
        history = pickle.load(file)

    losshistory(history, dir_path, plot_show=False)


plot = False

# plot loss history
if plot:
    replot_his()

# plot mnist digit
if plot:
    # data index range, start from 1 not 0
    start = 19
    end = 21
    data_range = np.arange(start, end+1)

    # load digit
    input_data, labels = set_input_data(start, end)

    # plot
    plot_digit(input_data, labels, data_range)
