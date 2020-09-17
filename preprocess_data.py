import os
import sys
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize



# Set some parameters
def get_X_Y(IMG_WIDTH, IMG_HEIGHT, TRAIN_PATH ):


    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    train_ids = next(os.walk(TRAIN_PATH))[1]
    # Get and resize train images and masks
    X = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    Y = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing images and masks... ')
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.resize(rgb, (IMG_HEIGHT, IMG_WIDTH), 1)
        im_gray = np.expand_dims(img_rgb, axis=-1)

        X[n] = im_gray
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH)), axis=-1)
            mask = np.maximum(mask, mask_)
        Y[n] = mask
    return X, Y
def split_data(X, Y):
    X_train = X[int(X.shape[0]*0.2):]
    Y_train = Y[int(Y.shape[0]*0.2):]
    X_test = X[:int(X.shape[0]*0.2)]
    Y_test = Y[:int(X.shape[0]*0.2)]
    return X_train, X_test, Y_train, Y_test
def check_plot_data(X, Y, ix = 0):
    p_X = imshow(X[ix])
    plt.show()
    p_Y = imshow(np.squeeze(Y[ix]))
    plt.show()
    print(Y[ix])
    print(Y[ix].shape)
    return p_X, p_Y