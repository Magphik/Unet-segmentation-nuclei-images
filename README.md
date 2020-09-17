# Unet-segmentation-nuclei-images
Using Unet- architecture network segmentates a large number of nuclei images from Kaggle.


The network architecture is shown in the Unet.png file.
The network consists of a contracting path and an expansive path, which gives it the u-shaped 
architecture. The contracting path is a typical convolutional network that consists of repeated 
application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling 
operation. During the contraction, the spatial information is reduced while feature information 
is increased. The expansive pathway combines the feature and spatial information through a 
sequence of up-convolutions and concatenations with high-resolution features from the contracting 
path.

Unet-segmentation contains next scripts:

Preprocessed_data.py - A script with a preprocessing function for a training set, 
converting a 3-channel image into a single-channel image and a function for splitting into 
training and validation samples.

Model.py - the described Unet model class, where the network is assembled

Train.py - a script for training a model for 30 epochs. After training, the obtained weights 
are saved and the graphs of accuracy and losses are displayed.

Predict_masks.py - A script that makes predictions based on the built model on test data
and writes them to a file. Also contains the function of outputting the received images from the file

Main.py - The main file that can be run to check the performance of the one that built 
the network will load the weights that we have trained, make a prediction and output them
and also calculate the estimate by the Sørensen-Dyes coefficient.

Evaluate_model.py - a script with a model estimate, the Sørensen-Dyes 
coefficient is taken as an estimate and the result.

The Sørensen–Dice coefficient (see below for other names) is a statistic used to gauge the 
similarity of two samples.

Data taken from link https://www.kaggle.com/c/data-science-bowl-2018/data
2018 Data Science Bowl
(Find the nuclei in divergent images to advance medical discovery)
This dataset contains a large number of segmented nuclei images. 
The images were acquired under a variety of conditions and vary in the cell type, 
magnification, and imaging modality (brightfield vs. fluorescence). 
The dataset is designed to challenge an algorithm's ability to generalize across these variations.

The requirements.txt file describes all the libraries and modules required for the project,
Run pip install -r requirements.txt (Python 2), or pip3 install -r requirements.txt (Python 3)

Author - Reveka Katerina
Mail - katerevekka@gmail.com
