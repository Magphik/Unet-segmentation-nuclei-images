import pathlib
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from skimage.io import imread, imshow
from scipy import ndimage
# Glob the training data and load a single image path
train_paths = pathlib.Path(r'.\data\stage1_train').glob('*/images/*.png')
train_sort = sorted([x for x in train_paths])
im_path = train_sort[7]
im = imageio.imread(str(im_path))
# Print the image dimensions
print('Original image shape: {}'.format(im.shape))

# Coerce the image into grayscale format (if not already)
im_gray = rgb2gray(im)
print('New image shape: {}'.format(im_gray.shape))
im_gray = np.expand_dims(im_gray, axis = -1)
print('New image shape for training: {}'.format(im_gray.shape))
cv2.imshow('Image_train', im_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
thresh_val = threshold_otsu(im_gray)
mask = np.where(im_gray > thresh_val, 1, 0)
# Make sure the larger portion of the mask is considered background
if np.sum(mask==0) < np.sum(mask==1):
    mask = np.where(mask, 0, 1)
plt.hist(mask.ravel(),256,[0,2])
plt.show()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here
histogram, bin_edges = np.histogram(im_gray, bins=256, range=(0, 1))
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()


labels, nlabels = ndimage.label(mask)

label_arrays = []
for label_num in range(1, nlabels + 1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)

print('There are {} separate components / objects detected.'.format(nlabels))
objects = []
plt.figure(figsize=(10,10))

for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
    cell = im_gray[label_coords]
    print(cell.shape)
    objects.append(cell)


    # Check if the label size is too small
    if np.product(cell.shape) < 10:
        print('Label {} is too small! Setting to 0.'.format(label_ind))
        mask = np.where(labels == label_ind + 1, 0, mask)
    else:
        plt.subplot(5, 5, label_ind+1)
        plt.imshow(np.squeeze(cell))
plt.show()
# print(objects)
# Regenerate the labels
labels, nlabels = ndimage.label(mask)
print('There are now {} separate components / objects detected.'.format(nlabels))
