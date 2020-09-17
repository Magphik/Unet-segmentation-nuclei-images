import os
import sys
import warnings
import cv2
from skimage import img_as_ubyte
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
import  numpy as np
from model import Unet
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = r'C:\Users\user\PycharmProjects\project_quantum\data\stage1_test/'
directory_for_saving = r'C:\Users\user\PycharmProjects\project_quantum\data\stage1_test_predicted/'
all_imgs = glob.glob(directory_for_saving+'*')

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
def make_predictions():

    test_ids = next(os.walk(TRAIN_PATH))[1]
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.resize(rgb, (IMG_HEIGHT, IMG_WIDTH), 1)
        im_gray = np.expand_dims(img_rgb, axis=-1)
        X_test[n] = im_gray
    print('Done!')
    unet = Unet()
    model = unet.build_model(IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH)
    checkpoint_path = "model_weights.h5"
    # model.save_weights(checkpoint_path)
    model.load_weights(checkpoint_path)
    predict_masks_X = model.predict(X_test)
def save_predicted_imgs():

    for i in range(predict_masks_X.shape[0]):
        rgb_pred = cv2.cvtColor(predict_masks_X[i], cv2.COLOR_GRAY2RGB)
        # imshow(rgb_pred)
        # plt.show()

        # Change the current directory
        # to specified directory
        os.chdir(directory_for_saving)

        # List files and directories
        print("Before saving image:")
        print(os.listdir(directory_for_saving))

        # Filename
        filename = str(i+1) + '.png'

        # Saving the image

        cv_image = img_as_ubyte(rgb_pred)
        cv2.imwrite(filename, cv_image)

        # List files and directories
        print("After saving image:")
        print(os.listdir(directory_for_saving))

        print('Successfully saved')
def display_pred_imgs():

    for i in range(len(all_imgs)):
        filename = directory_for_saving + str(i+1) + '.png'
        img = cv2.imread(filename)
        imshow(img)
        plt.show()
display_pred_imgs()