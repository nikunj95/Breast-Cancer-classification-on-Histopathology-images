Breast cancer is one of the common known cancer and IDC is the most common form of breast cancer. It is very important to identify and categorize breast cancer subtypes and methods which can do so automatically can not only save time but also help reduce errors identifying. As my interest in deep learning grows, it was only practical to use deep learning techniques to aid pathologist to help predict breast cancer. 

Special mentions to Adrian Rosebrock for posting a very useful tutorial for Breast classificaiton with Keras and Deep Learning. Find the tutorial here: https://www.pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/

Dataset used for this project - Breast Histopathology Images
Find the link to the dataset here: https://www.kaggle.com/paultimothymooney/breast-histopathology-images
It consists a total of 277,524 images belonging to two classes - positive and negative.
The number of positve images are 78,786 and the number of negative images are 198,738.

There are two files:
script_for_dataset.py - This script builds the dataset by splitting images in training, testing and validation sets.

train.py - This file defines the model and the layers that will be used for training. It then takes the datasets and starts training the model. After training is finished, it provides with the classificaiton report, accuracy, sensitivity and specificity.

Training is conducted using depthwise separable convolution.
The model acheives an accuracy of around 85%, sensitivity of ~ 85% and specificity of ~ 84%.
