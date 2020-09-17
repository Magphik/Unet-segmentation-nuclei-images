import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
import time
from skimage.io import imread, imshow, imread_collection, concatenate_images
from model import Unet
import preprocess_data
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = r'.\data\stage1_train/'
X, Y = preprocess_data.get_X_Y( IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, TRAIN_PATH = TRAIN_PATH)
X_train, X_test, Y_train, Y_test = preprocess_data.split_data(X, Y)
unet = Unet()
model = unet.build_model( IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
start_time = time.time()
history = model.fit(X_train,Y_train, batch_size=10,
                              epochs = 30, validation_split = 0.33,
                              verbose = 1        , callbacks=[learning_rate_reduction])
checkpoint_path = "model_weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
model.save_weights(checkpoint_path)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(model.summary())

