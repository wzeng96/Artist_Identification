# Import libraries
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

seed(1)
from tensorflow import set_random_seed

set_random_seed(1)

# %%
print(os.listdir("./input"))

# %%
artists = pd.read_csv('./input/artists.csv')
print(artists.shape())
print(artists.columns)

# ## Data Processing
artists.head()

# ### EDA

# Nationality
plt.figure(figsize=(10, 8))
plt.scatter(artists.name, artists.nationality)
plt.title('Scatter Plot of Artist Nationalities')
plt.xticks(rotation=90)
plt.show()

artists['nationality'].value_counts()

sns.set(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(20, 10))
graph = sns.countplot(ax=ax1, x='nationality', data=artists,
                      order=artists['nationality'].value_counts().index)

graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_title('Barplot for Artists Nationalities')
graph.set_xlabel('Nationality')
graph.set_ylabel('Count')
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x() + p.get_width() / 2., height + 0.1,
               artists['nationality'].value_counts()[i], ha="center")
    i += 1

# Art style/Genre
fig, ax2 = plt.subplots(figsize=(20, 10))
graph = sns.countplot(ax=ax2, x='genre', data=artists,
                      order=artists['genre'].value_counts().index)

graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
graph.set_title('Barplot for Artists Genre')
graph.set_xlabel('Genre')
graph.set_ylabel('Count')
i = 0
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x() + p.get_width() / 2., height + 0.1,
               artists['genre'].value_counts()[i], ha="center")
    i += 1

# %%
plt.figure(figsize=(10, 8))
plt.scatter(artists.genre, artists.nationality)
plt.title('Artist Nationalities vs. Genres')
plt.xticks(rotation=90)
plt.show()

# %%
# # of painting

plt.figure(figsize=(10, 8))
ax = plt.subplot()
bars = plt.barh(artists['name'], artists['paintings'], height=0.8)
annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(bar):
    x = bar.get_x() + bar.get_width() / 2.
    y = bar.get_y() + bar.get_height()
    annot.xy = (x, y)
    text = "({:.2g},{:.2g})".format(x, y)
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        for bar in bars:
            cont, ind = bar.contains(event)
            if cont:
                update_annot(bar)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    if vis:
        annot.set_visible(False)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.tight_layout()
plt.title('Number of Paintings for Each Artist')
plt.show()

#%%
# ### Data-Preprocessing for Model

# Sort artists by number of paintings
artists = artists.sort_values(by=['paintings'], ascending=False)

artists_top = artists[['name', 'paintings']].reset_index()
artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
print(artists_top.iloc[:, 1:4])

# %%
# Set class weights - assign higher weights to underrepresented classes
class_weights = artists_top['class_weight'].to_dict()
print(class_weights)

# %%
# updated_name = "Albrecht_Dürer".replace("_", " ")
updated_name = "Albrecht_Du╠êrer".replace("_", " ")
artists_top.iloc[4, 1] = updated_name
print(artists_top)

# %%
print(sum(artists_top['paintings']))

# Explore images of top artists
images_dir = '../input/images/images'
artists_dirs = os.listdir(images_dir)
artists_top_name = artists_top['name'].str.replace(' ', '_').values

# See if all directories exist
for name in artists_top_name:
    if os.path.exists(os.path.join(images_dir, name)):
        print("Found -->", os.path.join(images_dir, name))
    else:
        print("Did not find -->", os.path.join(images_dir, name))

# %%
# Print few random paintings
n = 5
fig, axes = plt.subplots(1, n, figsize=(20, 10))

for i in range(n):
    random_artist = random.choice(artists_top_name)
    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
    random_image_file = os.path.join(images_dir, random_artist, random_image)
    image = plt.imread(random_image_file)
    axes[i].imshow(image)
    axes[i].set_title("Artist: " + random_artist.replace('_', ' '))
    axes[i].axis('off')

plt.show()

#%%
# ### Random Forest Classifier

random_artist = artists_top_name
random_image = [os.listdir(os.path.join(images_dir, random_artist[i])) for i in range(0, len(random_artist))]
print(len(random_image[0]))


# random_image_file = [os.path.join(images_dir, random_artist[i], random_image[i]) for i in range(0, len(random_artist))]

# %%
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.resize(cv2.imread(os.path.join(folder, filename)), (224, 224))
            if img is not None:
                images.append(img)
    return images


# %%

folders = [os.path.join(images_dir, x) for x in random_artist]
all_images = [img for folder in folders for img in load_images_from_folder(folder)]

# %%
all_images = np.array(all_images)
print("all_images.shape:\n", all_images.shape)
print(artists_top.shape[0])

dataset_size = len(all_images)
TwoDim_dataset = all_images.reshape(dataset_size, -1)
print(TwoDim_dataset.shape)

# create label(artists)
# label = ['Vincent_van_Gogh']*877 + ['Edgar_Degas']*702
label = []
for i, j in zip(artists_top_name, artists_top['paintings']):
    label2 = [i] * j
    label.append(label2)
print(len(label[0]))

flat_label = [item for sublist in label for item in sublist]
len(flat_label)

X_train, X_test, y_train, y_test = model_selection.train_test_split(TwoDim_dataset, flat_label, train_size=0.80,
                                                                    test_size=0.20, random_state=101)

# RF
from sklearn.ensemble import RandomForestClassifier

# creating a RF classifier
clf = RandomForestClassifier(n_estimators=100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
from sklearn import metrics

print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

# %%
n_classes = artists_top.shape[0]
print("Classification Report: \n",
      metrics.classification_report(y_test, y_pred, target_names=artists_top_name.tolist()))

# %%
# ## Data Augmentation

# Augment data
batch_size = 16
train_input_shape = (224, 224, 3)
n_classes = artists_top.shape[0]

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1. / 255.,
                                   shear_range=5,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   )

train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                    )

valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                    )

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)

# %% [markdown]
# Print a random paintings and it's random augmented version
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

random_artist = random.choice(artists_top_name)
random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
random_image_file = os.path.join(images_dir, random_artist, random_image)

# Original image
image = plt.imread(random_image_file)
axes[0].imshow(image)
axes[0].set_title("An original Image of " + random_artist.replace('_', ' '))
axes[0].axis('off')

# Transformed image
aug_image = train_datagen.random_transform(image)
axes[1].imshow(aug_image)
axes[1].set_title("A transformed Image of " + random_artist.replace('_', ' '))
axes[1].axis('off')

plt.show()

# %% [markdown]
# ## Build Model
# CNN model
# Build model
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

# %%
lr_monitor = tf.keras.callbacks.LearningRateScheduler(
    lambda epochs: 1e-8 * 10 ** (epochs / 20))

model_check = ModelCheckpoint("keras-CNN-244-244.hdf5", monitor="val_accuracy", verbose=1, save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                           mode='auto', restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.2,
    min_lr=0.000001
)

epochs = 50

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=epochs,
                              callbacks=[learning_rate_reduction, model_check, early_stop],
                              class_weight=class_weights,
                              shuffle=True)

# %%
# get the values
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# show the results
epochs_range = range(epochs)

# %%
from matplotlib import gridspec

plt.rcParams.update({'font.size': 15})

fig = plt.figure(figsize=(12, 14))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

ax0 = plt.subplot(gs[0])
plt.plot(epochs_range, acc, label='Training')
plt.plot(epochs_range, val_acc, label='Validation')
plt.legend(loc='lower right')
plt.title('Accuracy and loss')
plt.ylabel("Accuracy")
plt.grid(True)

ax1 = plt.subplot(gs[1], sharex=ax0)
plt.plot(epochs_range, loss, label='Training')
plt.plot(epochs_range, val_loss, label='Validation')
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid(True)

plt.subplots_adjust(hspace=.0)
plt.show()

# %%
## Evaluation
# Prediction accuracy on train data
score = model.evaluate_generator(train_generator, verbose=1)
print("Prediction accuracy on train data =", score[1])

# Prediction accuracy on testing data
score = model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on CV data =", score[1])

# %%
# Classification report and confusion matrix
tick_labels = artists_top_name.tolist()


def showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):
    # Loop on each generator batch and predict
    y_pred, y_true = [], []
    for i in range(STEP_SIZE_VALID):
        (X, y) = next(valid_generator)
        y_pred.append(model.predict(X))
        y_true.append(y)

    # Create a flat list for y_true and y_pred
    y_pred = [subresult for result in y_pred for subresult in result]
    y_true = [subresult for result in y_true for subresult in result]

    # Update Truth vector based on argmax
    y_true = np.argmax(y_true, axis=1)
    y_true = np.asarray(y_true).ravel()

    # Update Prediction vector based on argmax
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.asarray(y_pred).ravel()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(50, 30))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", square=True, cbar=False,
                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.show()

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=np.arange(n_classes), target_names=artists_top_name.tolist()))


showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID)

# %%
# Prediction

n = 5
fig, axes = plt.subplots(1, n, figsize=(25, 10))

for i in range(n):
    random_artist = random.choice(artists_top_name)
    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
    random_image_file = os.path.join(images_dir, random_artist, random_image)

    # Original image

    test_image = image.load_img(random_image_file, target_size=(train_input_shape[0:2]))

    # Predict artist
    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)

    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())

    title = "Actual artist = {}\nPredicted artist = {}\nPrediction probability = {:.2f} %" \
        .format(random_artist.replace('_', ' '), labels[prediction_idx].replace('_', ' '),
                prediction_probability * 100)

    # Print image
    axes[i].imshow(plt.imread(random_image_file))
    axes[i].set_title(title)
    axes[i].axis('off')

plt.show()

# %%
# ### ResNet 50

# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

for layer in base_model.layers:
    layer.trainable = True

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"jupyter":{"outputs_hidden":false}}
# Add layers at the end
X = base_model.output
X = Flatten()(X)

X = Dense(512, kernel_initializer='he_uniform')(X)
# X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

X = Dense(16, kernel_initializer='he_uniform')(X)
# X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output = Dense(n_classes, activation='softmax')(X)

model = Model(inputs=base_model.input, outputs=output)

# %%
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# %%
n_epoch = 10

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='auto')

# %%
# Train the model - all layers
history1 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                               validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                               epochs=n_epoch,
                               shuffle=True,
                               verbose=1,
                               callbacks=[reduce_lr],
                               use_multiprocessing=True,
                               workers=16,
                               class_weight=class_weights
                               )

# Freeze core ResNet layers and train again
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[:50]:
    layer.trainable = True

optimizer = Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

n_epoch = 50
history2 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                               validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                               epochs=n_epoch,
                               shuffle=True,
                               verbose=1,
                               callbacks=[reduce_lr, early_stop],
                               use_multiprocessing=True,
                               workers=16,
                               class_weight=class_weights
                               )

#
# ## Training graph

# %%
# Merge history1 and history2
history = {}
history['loss'] = history1.history['loss'] + history2.history['loss']
history['acc'] = history1.history['acc'] + history2.history['acc']
history['val_loss'] = history1.history['val_loss'] + history2.history['val_loss']
history['val_acc'] = history1.history['val_acc'] + history2.history['val_acc']
history['lr'] = history1.history['lr'] + history2.history['lr']


# %%
# Plot the training graph
def plot_training(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='best')

    plt.show()


plot_training(history)

# %% [markdown]
# ## Evaluate performance

# Prediction accuracy on train data
score = model.evaluate_generator(train_generator, verbose=1)
print("Prediction accuracy on train data =", score[1])

# %%
# Prediction accuracy on CV data
score = model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on CV data =", score[1])

# %%
# ## Confusion Matrix. Look at the style of the artists which the model thinks are almost similar.

# Classification report and confusion matrix

tick_labels = artists_top_name.tolist()


def showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):
    # Loop on each generator batch and predict
    y_pred, y_true = [], []
    for i in range(STEP_SIZE_VALID):
        (X, y) = next(valid_generator)
        y_pred.append(model.predict(X))
        y_true.append(y)

    # Create a flat list for y_true and y_pred
    y_pred = [subresult for result in y_pred for subresult in result]
    y_true = [subresult for result in y_true for subresult in result]

    # Update Truth vector based on argmax
    y_true = np.argmax(y_true, axis=1)
    y_true = np.asarray(y_true).ravel()

    # Update Prediction vector based on argmax
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.asarray(y_pred).ravel()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(50, 30))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", square=True, cbar=False,
                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.show()

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=np.arange(n_classes), target_names=artists_top_name.tolist()))


showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID)

# %%
# # Evaluate performance by predicting on random images from dataset

# Prediction
n = 5
fig, axes = plt.subplots(1, n, figsize=(25, 10))

for i in range(n):
    random_artist = random.choice(artists_top_name)
    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
    random_image_file = os.path.join(images_dir, random_artist, random_image)

    # Original image

    test_image = image.load_img(random_image_file, target_size=(train_input_shape[0:2]))

    # Predict artist
    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)

    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())

    # print("Actual artist =", random_artist.replace('_', ' '))
    # print("Predicted artist =", labels[prediction_idx].replace('_', ' '))
    # print("Prediction probability =", prediction_probability*100, "%")

    title = "Actual artist = {}\nPredicted artist = {}\nPrediction probability = {:.2f} %" \
        .format(random_artist.replace('_', ' '), labels[prediction_idx].replace('_', ' '),
                prediction_probability * 100)

    # Print image
    axes[i].imshow(plt.imread(random_image_file))
    axes[i].set_title(title)
    axes[i].axis('off')

plt.show()

# %%
# Predict from web - this is an image of Titian.
url = 'https://www.gpsmycity.com/img/gd/2081.jpg'

web_image = imageio.imread(url)
web_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )
web_image = image.img_to_array(web_image)
web_image /= 255.
web_image = np.expand_dims(web_image, axis=0)

prediction = model.predict(web_image)
prediction_probability = np.amax(prediction)
prediction_idx = np.argmax(prediction)

print("Predicted artist =", labels[prediction_idx].replace('_', ' '))
print("Prediction probability =", prediction_probability * 100, "%")

plt.imshow(imageio.imread(url))
plt.axis('off')
plt.show()

print(model.summary())
