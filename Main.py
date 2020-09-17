import random
import numpy as np
import matplotlib.pyplot as plt
from model import Unet
from skimage.io import imread, imshow, imread_collection, concatenate_images
import evaluate_model

import cv2

import preprocess_data
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '.\data\stage1_train/'
X, Y = preprocess_data.get_X_Y(IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, TRAIN_PATH = TRAIN_PATH)
X_train, X_test, Y_train, Y_test = preprocess_data.split_data(X, Y)
img_avg = np.zeros((Y.shape[0], IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
img_objects = np.zeros((len(Y), 1), dtype=np.int)
for i in range(len(Y)):
    img_avg += np.asarray(Y[i])
    img_objects[i] =  len(np.where(Y[i] > 0))

img_avg = img_avg/ len(Y)
print(img_objects)

unet = Unet()
model = unet.build_model(IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH)

checkpoint_path = "model_weights.h5"
model.load_weights(checkpoint_path)

# Predict on train, val and test
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)
print(preds_test, type(preds_test), preds_test.shape)
# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
print(preds_train_t, preds_val_t, preds_test_t)
# Create list of upsampled test masks
preds_test_upsampled = []
ix = random.randint(0, len(preds_test_t))

rgb1 = cv2.cvtColor(X_train[ix], cv2.COLOR_GRAY2RGB)
imshow(rgb1)
plt.show()
rgb2 = cv2.cvtColor(np.float32(Y_train[ix]), cv2.COLOR_GRAY2RGB)
imshow(rgb2)
plt.show()
rgb3 = cv2.cvtColor(np.float32(preds_test_t[ix]), cv2.COLOR_GRAY2RGB)
imshow(rgb3)
plt.show()

# closing all open windows
preds_test1 = model.predict(X_test, verbose=1)
dice_all_pred = evaluate_model.calc_dice(Y_test, preds_test1)
print(dice_all_pred)
print(preds_test1.shape, type(preds_test1[5]))
preds_test2 = np.zeros(preds_test1.shape)

preds_test1_t = (preds_test1 > 0.5).astype(np.uint8)
dice_all = evaluate_model.calc_dice(Y_test, preds_test1_t)
print(dice_all)
columns=['Predicts']


for i in range(preds_test1_t.shape[0]):
    dice_each = evaluate_model.calc_dice(Y_test[i], preds_test1_t[i])
    print(dice_each)
