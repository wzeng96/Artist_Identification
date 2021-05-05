# Author Identification of Works of Art: Convolutional Neural Networks Modeling
### Capstone Project
### The George Washington University

Welcome! This project is about identifying artist based on a piece of artwork, with no other information given. The three main methods are Random Forest Classifier, Convolutional Neural Network (CNN), and Residual Network 50 (ResNet 50). By comparing the accuracy scores of the three models, we would choose the most optimal model for future uses.

### Introduction
Author identification of artwork identifies the artist wholly based on the artwork itself, with no other information given. With the increasing need to digitalize images in order to save the art pieces better, artwork identification has become more and more critical. Art historians begin to use Data Visualization and Deep Learning techniques to help recognize pieces of artwork that either belongs to well-known artists, undiscovered artists, or random artists from antique stores (C. R. Johnson et al., 2008). 

The main methods for this project are Convolutional Neural Networks (CNN) and ResNet 50 models. ResNet is the shortcut for Residual Network, and 50 means the pre-trained 50 layers. We also include a Random Forest Classifier because Random Forest also shows good image classification problems, especially in the Biomedical field. Research has shown the possible improvements for multi-class classification in Random Forest (Chaudhary, Kolhe and Kamal. 2016). Therefore, we apply the Random Forest model to check whether it can perform well in artwork recognitions. CNN model is a fundamental deep learning model for image classification. One of the main reasons is that CNN effectively reduces the number of parameters without losing the quality of models. Since each pixel in an image is considered a feature, CNN is suitable for this high dimensionality of images. We consider using the ResNet 50 model because this Deep Learning model is trained on a different task than the task at hand. This winner of the ImageNet detection in 2015 (K. He, X. Zhang, S. Ren and J. Sun. 2016) allows us to successfully train extremely deep neural networks with more than 150 layers. We believe using the ResNet 50 model will significantly improve our accuracy for the project.

The goal of this project is to train Random Forest, Convolutional Neural Networks (CNN), and ResNet 50 models to make accurate predictions on the authority of the given art piece. 

### Coding
#### Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import *
import seaborn as sns
import os
from tqdm import tqdm, tqdm_notebook
import random
import cv2
from keras.preprocessing import *
import imageio
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed
from tensorflow import set_random_seed
```

#### Creation of CNN Model
```
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), input_shape=train_input_shape, activation=tf.keras.activations.relu,
                           padding='same'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, padding='same'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # padding = 'same'
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')

])

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=tf.keras.optimizers.Adam(lr=3e-4))
```
              
#### Creation of ResNet 50 Model
```
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

for layer in base_model.layers:
    layer.trainable = True

X = base_model.output
X = Flatten()(X)

X = Dense(512, kernel_initializer='he_uniform')(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

X = Dense(16, kernel_initializer='he_uniform')(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output = Dense(n_classes, activation='softmax')(X)

model = Model(inputs=base_model.input, outputs=output)

optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
```
              
### Conclusions

A comparison table with all the accuracy scores for the three models.

|                      |     Full Dataset    |                    |     Artists with more than 200 paintings    |                    |

|----------------------|---------------------|--------------------|---------------------------------------------|--------------------|
|                      |     Training Set    |     Testing Set    |     Training Set                            |     Testing Set    |
|     Random Forest    |     ———————         |     27%            |     ———————                                 |     51%            |
|     CNN              |     72.63%          |     37.25%         |     60.68%                                  |     48.42%         |
|     ResNet 50        |     75.60%          |     53.39%         |     99.94%                                  |     85.73%         |

The project manages to collect 8,446 art pieces from 50 famous western artists from 1226 to 1989. By applying Random Forest Classifier, CNN, and ResNet 50 models to the two datasets, we successfully train the models to learn the characteristics of different artists and predict the authority of a piece of artwork based on the painting itself, with no other information given. The high accuracy, 85.73% in the testing set, of the ResNet 50 model shows that convolutional neural networks are powerful tools for image recognition and image classification. We have also shown that Random Forest Classifier is a better choice when the data is balanced than CNN in image classification problems. Although many other models work well with image classification or image recognition, ResNet 50 is relatively less time-consuming and requires fewer computing resources compared to other pre-trained artificial neural networks. Eventually, ResNet 50 could fit well into the help of identifying an unknown piece of artwork, and figure out if it comes from a famous artist, an antique store, someone’s attic, or an undiscovered artist. 

