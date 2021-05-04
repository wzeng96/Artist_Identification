{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [markdown]\n# ## Read data\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Import libraries\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport json\nimport os\nfrom tqdm import tqdm, tqdm_notebook\nimport random\n\nimport tensorflow as tf\nfrom tensorflow.keras.models import Sequential, Model\nfrom tensorflow.keras.layers import *\nfrom tensorflow.keras.optimizers import *\nfrom tensorflow.keras.applications import *\nfrom tensorflow.keras.callbacks import *\nfrom tensorflow.keras.initializers import *\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\n\nfrom numpy.random import seed\nseed(1)\nfrom tensorflow import set_random_seed\nset_random_seed(1)\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\nprint(os.listdir(\"../input\"))\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\nartists = pd.read_csv('../input/artists.csv')\nartists.shape\nprint(artists.columns)\n\n# %% [markdown]\n# ## Data Processing\n\n# %% [code]\nartists.head()\n\n# %% [markdown]\n# ### EDA\n\n# %% [code]\n# Nationality\nplt.figure(figsize=(10,8))\nplt.scatter(artists.name, artists.nationality)\nplt.title('Scatter Plot of Artist Nationalities')\nplt.xticks(rotation=90)\nplt.show()\n\n# %% [code]\nartists['nationality'].value_counts()\n\n# %% [code]\nimport seaborn as sns\nsns.set(style=\"whitegrid\")\n\nfig, ax1 = plt.subplots(figsize=(20,10))\ngraph = sns.countplot(ax=ax1,x='nationality', data=artists,\n                     order = artists['nationality'].value_counts().index)\n\ngraph.set_xticklabels(graph.get_xticklabels(),rotation=90)\ngraph.set_title('Barplot for Artists Nationalities')\ngraph.set_xlabel('Nationality')\ngraph.set_ylabel('Count')\ni=0\nfor p in graph.patches:\n    height = p.get_height()\n    graph.text(p.get_x()+p.get_width()/2., height + 0.1,\n        artists['nationality'].value_counts()[i],ha=\"center\")\n    i += 1\n\n# %% [code]\n# Art style/Genre\nfig, ax2 = plt.subplots(figsize=(20,10))\ngraph = sns.countplot(ax=ax2,x='genre', data=artists,\n                     order = artists['genre'].value_counts().index)\n\ngraph.set_xticklabels(graph.get_xticklabels(),rotation=90)\ngraph.set_title('Barplot for Artists Genre')\ngraph.set_xlabel('Genre')\ngraph.set_ylabel('Count')\ni=0\nfor p in graph.patches:\n    height = p.get_height()\n    graph.text(p.get_x()+p.get_width()/2., height + 0.1,\n        artists['genre'].value_counts()[i],ha=\"center\")\n    i += 1\n\n# %% [code]\nplt.figure(figsize=(10,8))\nplt.scatter(artists.genre, artists.nationality)\nplt.title('Artist Nationalities vs. Genres')\nplt.xticks(rotation=90)\nplt.show()\n\n# %% [code]\n# # of painting\n\nplt.figure(figsize=(10,8))\nax=plt.subplot()\nbars = plt.barh(artists['name'], artists['paintings'], height=0.8)\nannot = ax.annotate(\"\", xy=(0,0), xytext=(-20,20),textcoords=\"offset points\",\n                    bbox=dict(boxstyle=\"round\", fc=\"black\", ec=\"b\", lw=2),\n                    arrowprops=dict(arrowstyle=\"->\"))\nannot.set_visible(False)\n\ndef update_annot(bar):\n    x = bar.get_x()+bar.get_width()/2.\n    y = bar.get_y()+bar.get_height()\n    annot.xy = (x,y)\n    text = \"({:.2g},{:.2g})\".format( x,y )\n    annot.set_text(text)\n    annot.get_bbox_patch().set_alpha(0.4)\n\n\ndef hover(event):\n    vis = annot.get_visible()\n    if event.inaxes == ax:\n        for bar in bars:\n            cont, ind = bar.contains(event)\n            if cont:\n                update_annot(bar)\n                annot.set_visible(True)\n                fig.canvas.draw_idle()\n                return\n    if vis:\n        annot.set_visible(False)\n        fig.canvas.draw_idle()\n\nfig.canvas.mpl_connect(\"motion_notify_event\", hover)\n\nplt.tight_layout()\nplt.title('Number of Paintings for Each Artist')\nplt.show()\n\n\n# %% [markdown]\n# ### Data-Preprocessing for Model\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Sort artists by number of paintings\nartists = artists.sort_values(by=['paintings'], ascending=False)\n\nartists_top = artists[['name', 'paintings']].reset_index()\nartists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)\nartists_top.iloc[:, 1:4]\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Set class weights - assign higher weights to underrepresented classes\nclass_weights = artists_top['class_weight'].to_dict()\nclass_weights\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# updated_name = \"Albrecht_Dürer\".replace(\"_\", \" \")\nupdated_name = \"Albrecht_Du╠êrer\".replace(\"_\", \" \")\nartists_top.iloc[4, 1] = updated_name \nprint(artists_top)\n\n# %% [code]\nprint(sum(artists_top['paintings']))\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Explore images of top artists\nimages_dir = '../input/images/images'\nartists_dirs = os.listdir(images_dir)\nartists_top_name = artists_top['name'].str.replace(' ', '_').values\n\n# See if all directories exist\nfor name in artists_top_name:\n    if os.path.exists(os.path.join(images_dir, name)):\n        print(\"Found -->\", os.path.join(images_dir, name))\n    else:\n        print(\"Did not find -->\", os.path.join(images_dir, name))\n\n# %% [markdown]\n# ### Print few random paintings\n\n# %% [code] {\"_kg_hide-input\":true}\n# Print few random paintings\nn = 5\nfig, axes = plt.subplots(1, n, figsize=(20,10))\n\nfor i in range(n):\n    random_artist = random.choice(artists_top_name)\n    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))\n    random_image_file = os.path.join(images_dir, random_artist, random_image)\n    image = plt.imread(random_image_file)\n    axes[i].imshow(image)\n    axes[i].set_title(\"Artist: \" + random_artist.replace('_', ' '))\n    axes[i].axis('off')\n\nplt.show()\n\n# %% [markdown]\n# ### Random Forest Classifier\n\n# %% [code]\nfrom sklearn import svm\nimport sklearn.model_selection as model_selection\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import f1_score\n\n# %% [code]\n# random_artist = random.choice(artists_top_name)\n# random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))\n# random_image_file = os.path.join(images_dir, random_artist, random_image)\nrandom_artist = artists_top_name\nrandom_image = [os.listdir(os.path.join(images_dir, random_artist[i])) for i in range(0, len(random_artist))]\nprint(len(random_image[0]))\n# random_image_file = [os.path.join(images_dir, random_artist[i], random_image[i]) for i in range(0, len(random_artist))]\n\n\n# %% [code]\ndef load_images_from_folder(folder):\n    images = []\n    for filename in os.listdir(folder):\n        if filename.endswith(\".jpg\"):\n            img = cv2.resize(cv2.imread(os.path.join(folder, filename)),(224,224))\n            if img is not None:\n                images.append(img)\n    return images\n\n# %% [code]\nimport cv2\nfolders = [os.path.join(images_dir, x) for x in random_artist]\nall_images = [img for folder in folders for img in load_images_from_folder(folder)]\n\n# %% [code]\nall_images = np.array(all_images)\nprint(\"all_images.shape:\\n\", all_images.shape)\nprint(artists_top.shape[0])\n\n# %% [code]\ndataset_size = len(all_images)\nTwoDim_dataset = all_images.reshape(dataset_size,-1)\nprint(TwoDim_dataset.shape)\n\n# %% [code]\n# create label(artists)\n# label = ['Vincent_van_Gogh']*877 + ['Edgar_Degas']*702\nlabel = []\nfor i,j in zip(artists_top_name, artists_top['paintings']):\n    label2 = [i] * j\n    label.append(label2)\nprint(len(label[0]))\n\n# %% [code]\nflat_label = [item for sublist in label for item in sublist]\nlen(flat_label)\n\n# %% [code]\nX_train, X_test, y_train, y_test = model_selection.train_test_split(TwoDim_dataset, flat_label, train_size=0.80, test_size=0.20, random_state=101)\n\n# %% [code]\n# RF\nfrom sklearn.ensemble import RandomForestClassifier\n# creating a RF classifier\nclf = RandomForestClassifier(n_estimators = 100)\n\n# Training the model on the training dataset\n# fit function is used to train the model using the training sets as parameters\nclf.fit(X_train, y_train)\n\n# performing predictions on the test dataset\ny_pred = clf.predict(X_test)\n\n# metrics are used to find accuracy or error\nfrom sklearn import metrics\nprint()\n\n# using metrics module for accuracy calculation\nprint(\"ACCURACY OF THE MODEL: \", metrics.accuracy_score(y_test, y_pred))\n\n\n# %% [code]\nn_classes = artists_top.shape[0]\nprint(\"Classification Report: \\n\", metrics.classification_report(y_test, y_pred, target_names=artists_top_name.tolist()))\n\n# %% [markdown]\n# ## Data Augmentation\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Augment data\nbatch_size = 16\ntrain_input_shape = (224, 224, 3)\nn_classes = artists_top.shape[0]\n\ntrain_datagen = ImageDataGenerator(validation_split=0.2,\n                                   rescale=1./255.,\n                                   shear_range=5,\n                                   horizontal_flip=True,\n                                   vertical_flip=True,\n                                  )\n\ntrain_generator = train_datagen.flow_from_directory(directory=images_dir,\n                                                    class_mode='categorical',\n                                                    target_size=train_input_shape[0:2],\n                                                    batch_size=batch_size,\n                                                    subset=\"training\",\n                                                    shuffle=True,\n                                                    classes=artists_top_name.tolist()\n                                                   )\n\nvalid_generator = train_datagen.flow_from_directory(directory=images_dir,\n                                                    class_mode='categorical',\n                                                    target_size=train_input_shape[0:2],\n                                                    batch_size=batch_size,\n                                                    subset=\"validation\",\n                                                    shuffle=True,\n                                                    classes=artists_top_name.tolist()\n                                                   )\n\nSTEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size\nSTEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size\nprint(\"Total number of batches =\", STEP_SIZE_TRAIN, \"and\", STEP_SIZE_VALID)\n\n# %% [markdown]\n# ### Print a random paintings and it's random augmented version\n\n# %% [code] {\"_kg_hide-input\":true}\n# Print a random paintings and it's random augmented version\nfig, axes = plt.subplots(1, 2, figsize=(20,10))\n\nrandom_artist = random.choice(artists_top_name)\nrandom_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))\nrandom_image_file = os.path.join(images_dir, random_artist, random_image)\n\n# Original image\nimage = plt.imread(random_image_file)\naxes[0].imshow(image)\naxes[0].set_title(\"An original Image of \" + random_artist.replace('_', ' '))\naxes[0].axis('off')\n\n# Transformed image\naug_image = train_datagen.random_transform(image)\naxes[1].imshow(aug_image)\naxes[1].set_title(\"A transformed Image of \" + random_artist.replace('_', ' '))\naxes[1].axis('off')\n\nplt.show()\n\n# %% [markdown]\n# ## Build Model\n\n# %% [code]\n# CNN model\n# Build model\nmodel = tf.keras.Sequential([\n\n    tf.keras.layers.Conv2D(32, (3, 3), input_shape=train_input_shape, activation=tf.keras.activations.relu, padding = 'same'),\n    tf.keras.layers.MaxPool2D(2, 2),\n    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.relu, padding = 'same'),\n    tf.keras.layers.MaxPool2D(2, 2),\n    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'), # padding = 'same'\n    tf.keras.layers.MaxPool2D(2, 2),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Flatten(),\n    tf.keras.layers.Dense(256, activation='relu'),\n    tf.keras.layers.Dense(n_classes, activation = 'softmax')\n\n])\n\nmodel.compile(loss='categorical_crossentropy',\n              metrics=['accuracy'],\n              optimizer=tf.keras.optimizers.Adam(lr=3e-4))\n\n# %% [code]\nlr_monitor = tf.keras.callbacks.LearningRateScheduler(\n    lambda epochs: 1e-8 * 10 ** (epochs / 20))\n\nmodel_check = ModelCheckpoint(\"keras-CNN-244-244.hdf5\", monitor=\"val_accuracy\", verbose=1, save_best_only=True)\n\nearly_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, \n                           mode='auto', restore_best_weights=True)\n\nlearning_rate_reduction = ReduceLROnPlateau(\n    monitor='val_accuracy',\n    patience=3,\n    verbose=1,\n    factor=0.2,\n    min_lr=0.000001\n)\n\nepochs = 50\n\n# fits the model on batches with real-time data augmentation:\nhistory = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,\n                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,\n                              epochs=epochs,\n                              callbacks=[learning_rate_reduction, model_check, early_stop],\n                              class_weight=class_weights,\n                              shuffle=True)\n\n# %% [code]\n# get the values\nacc      = history.history['acc']\nval_acc  = history.history['val_acc']\nloss     = history.history['loss']\nval_loss = history.history['val_loss']\n\n\n# show the results\nepochs_range = range(epochs)\n\n\n\n# %% [code]\nfrom matplotlib import gridspec\nplt.rcParams.update({'font.size': 15})\n\nfig = plt.figure(figsize=(12, 14))\ngs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) \n\nax0 = plt.subplot(gs[0])\nplt.plot(epochs_range, acc, label='Training')\nplt.plot(epochs_range, val_acc, label='Validation')\nplt.legend(loc='lower right')\nplt.title('Accuracy and loss')\nplt.ylabel(\"Accuracy\")\nplt.grid(True)\n\nax1 = plt.subplot(gs[1], sharex = ax0)\nplt.plot(epochs_range, loss, label='Training')\nplt.plot(epochs_range, val_loss, label='Validation')\nplt.ylabel(\"Loss\")\nplt.xlabel(\"Epochs\")\nplt.grid(True)\n\nplt.subplots_adjust(hspace=.0)\nplt.show()\n\n# %% [code]\n## Evaluation\n# Prediction accuracy on train data\nscore = model.evaluate_generator(train_generator, verbose=1)\nprint(\"Prediction accuracy on train data =\", score[1])\n\n# Prediction accuracy on testing data\nscore = model.evaluate_generator(valid_generator, verbose=1)\nprint(\"Prediction accuracy on CV data =\", score[1])\n\n# %% [code]\n# Classification report and confusion matrix\nfrom sklearn.metrics import *\nimport seaborn as sns\n\ntick_labels = artists_top_name.tolist()\n\ndef showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):\n    # Loop on each generator batch and predict\n    y_pred, y_true = [], []\n    for i in range(STEP_SIZE_VALID):\n        (X,y) = next(valid_generator)\n        y_pred.append(model.predict(X))\n        y_true.append(y)\n    \n    # Create a flat list for y_true and y_pred\n    y_pred = [subresult for result in y_pred for subresult in result]\n    y_true = [subresult for result in y_true for subresult in result]\n    \n    # Update Truth vector based on argmax\n    y_true = np.argmax(y_true, axis=1)\n    y_true = np.asarray(y_true).ravel()\n    \n    # Update Prediction vector based on argmax\n    y_pred = np.argmax(y_pred, axis=1)\n    y_pred = np.asarray(y_pred).ravel()\n    \n    # Confusion Matrix\n    fig, ax = plt.subplots(figsize=(50,30))\n    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))\n    conf_matrix = conf_matrix/np.sum(conf_matrix, axis=1)\n    sns.heatmap(conf_matrix, annot=True, fmt=\".2f\", square=True, cbar=False, \n                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,\n                ax=ax)\n    ax.set_ylabel('Actual')\n    ax.set_xlabel('Predicted')\n    ax.set_title('Confusion Matrix')\n    plt.show()\n    \n    print('Classification Report:')\n    print(classification_report(y_true, y_pred, labels=np.arange(n_classes), target_names=artists_top_name.tolist()))\n\nshowClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID)\n\n# %% [code]\n# Prediction\nfrom keras.preprocessing import *\n\nn = 5\nfig, axes = plt.subplots(1, n, figsize=(25,10))\n\nfor i in range(n):\n    random_artist = random.choice(artists_top_name)\n    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))\n    random_image_file = os.path.join(images_dir, random_artist, random_image)\n\n    # Original image\n\n    test_image = image.load_img(random_image_file, target_size=(train_input_shape[0:2]))\n\n    # Predict artist\n    test_image = image.img_to_array(test_image)\n    test_image /= 255.\n    test_image = np.expand_dims(test_image, axis=0)\n\n    prediction = model.predict(test_image)\n    prediction_probability = np.amax(prediction)\n    prediction_idx = np.argmax(prediction)\n\n    labels = train_generator.class_indices\n    labels = dict((v,k) for k,v in labels.items())\n\n    title = \"Actual artist = {}\\nPredicted artist = {}\\nPrediction probability = {:.2f} %\" \\\n                .format(random_artist.replace('_', ' '), labels[prediction_idx].replace('_', ' '),\n                        prediction_probability*100)\n\n    # Print image\n    axes[i].imshow(plt.imread(random_image_file))\n    axes[i].set_title(title)\n    axes[i].axis('off')\n\nplt.show()\n\n# %% [markdown]\n# ### ResNet 50\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Load pre-trained model\nbase_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)\n\nfor layer in base_model.layers:\n    layer.trainable = True\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Add layers at the end\nX = base_model.output\nX = Flatten()(X)\n\nX = Dense(512, kernel_initializer='he_uniform')(X)\n#X = Dropout(0.5)(X)\nX = BatchNormalization()(X)\nX = Activation('relu')(X)\n\nX = Dense(16, kernel_initializer='he_uniform')(X)\n#X = Dropout(0.5)(X)\nX = BatchNormalization()(X)\nX = Activation('relu')(X)\n\noutput = Dense(n_classes, activation='softmax')(X)\n\nmodel = Model(inputs=base_model.input, outputs=output)\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\noptimizer = Adam(lr=0.0001)\nmodel.compile(loss='categorical_crossentropy',\n              optimizer=optimizer, \n              metrics=['accuracy'])\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\nn_epoch = 10\n\nearly_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, \n                           mode='auto', restore_best_weights=True)\n\nreduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, \n                              verbose=1, mode='auto')\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Train the model - all layers\nhistory1 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,\n                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,\n                              epochs=n_epoch,\n                              shuffle=True,\n                              verbose=1,\n                              callbacks=[reduce_lr],\n                              use_multiprocessing=True,\n                              workers=16,\n                              class_weight=class_weights\n                             )\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Freeze core ResNet layers and train again \nfor layer in model.layers:\n    layer.trainable = False\n\nfor layer in model.layers[:50]:\n    layer.trainable = True\n\noptimizer = Adam(lr=0.0001)\n\nmodel.compile(loss='categorical_crossentropy',\n              optimizer=optimizer, \n              metrics=['accuracy'])\n\nn_epoch = 50\nhistory2 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,\n                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,\n                              epochs=n_epoch,\n                              shuffle=True,\n                              verbose=1,\n                              callbacks=[reduce_lr, early_stop],\n                              use_multiprocessing=True,\n                              workers=16,\n                              class_weight=class_weights\n                             )\n\n# %% [markdown]\n# ## Training graph\n\n# %% [code] {\"_kg_hide-input\":true,\"_kg_hide-output\":true}\n# Merge history1 and history2\nhistory = {}\nhistory['loss'] = history1.history['loss'] + history2.history['loss']\nhistory['acc'] = history1.history['acc'] + history2.history['acc']\nhistory['val_loss'] = history1.history['val_loss'] + history2.history['val_loss']\nhistory['val_acc'] = history1.history['val_acc'] + history2.history['val_acc']\nhistory['lr'] = history1.history['lr'] + history2.history['lr']\n\n# %% [code] {\"_kg_hide-input\":true}\n# Plot the training graph\ndef plot_training(history):\n    acc = history['acc']\n    val_acc = history['val_acc']\n    loss = history['loss']\n    val_loss = history['val_loss']\n    epochs = range(len(acc))\n\n    fig, axes = plt.subplots(1, 2, figsize=(15,5))\n    \n    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')\n    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')\n    axes[0].set_title('Training and Validation Accuracy')\n    axes[0].legend(loc='best')\n\n    axes[1].plot(epochs, loss, 'r-', label='Training Loss')\n    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')\n    axes[1].set_title('Training and Validation Loss')\n    axes[1].legend(loc='best')\n    \n    plt.show()\n    \nplot_training(history)\n\n# %% [markdown]\n# ## Evaluate performance\n\n# %% [code] {\"_kg_hide-input\":true}\n# Prediction accuracy on train data\nscore = model.evaluate_generator(train_generator, verbose=1)\nprint(\"Prediction accuracy on train data =\", score[1])\n\n# %% [code] {\"_kg_hide-input\":true}\n# Prediction accuracy on CV data\nscore = model.evaluate_generator(valid_generator, verbose=1)\nprint(\"Prediction accuracy on CV data =\", score[1])\n\n# %% [markdown]\n# ## Confusion Matrix. Look at the style of the artists which the model thinks are almost similar. \n\n# %% [code] {\"_kg_hide-input\":true}\n# Classification report and confusion matrix\nfrom sklearn.metrics import *\nimport seaborn as sns\n\ntick_labels = artists_top_name.tolist()\n\ndef showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):\n    # Loop on each generator batch and predict\n    y_pred, y_true = [], []\n    for i in range(STEP_SIZE_VALID):\n        (X,y) = next(valid_generator)\n        y_pred.append(model.predict(X))\n        y_true.append(y)\n    \n    # Create a flat list for y_true and y_pred\n    y_pred = [subresult for result in y_pred for subresult in result]\n    y_true = [subresult for result in y_true for subresult in result]\n    \n    # Update Truth vector based on argmax\n    y_true = np.argmax(y_true, axis=1)\n    y_true = np.asarray(y_true).ravel()\n    \n    # Update Prediction vector based on argmax\n    y_pred = np.argmax(y_pred, axis=1)\n    y_pred = np.asarray(y_pred).ravel()\n    \n    # Confusion Matrix\n    fig, ax = plt.subplots(figsize=(50,30))\n    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))\n    conf_matrix = conf_matrix/np.sum(conf_matrix, axis=1)\n    sns.heatmap(conf_matrix, annot=True, fmt=\".2f\", square=True, cbar=False, \n                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,\n                ax=ax)\n    ax.set_ylabel('Actual')\n    ax.set_xlabel('Predicted')\n    ax.set_title('Confusion Matrix')\n    plt.show()\n    \n    print('Classification Report:')\n    print(classification_report(y_true, y_pred, labels=np.arange(n_classes), target_names=artists_top_name.tolist()))\n\nshowClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID)\n\n# %% [markdown]\n# # Evaluate performance by predicting on random images from dataset\n\n# %% [code] {\"_kg_hide-input\":true}\n# Prediction\nfrom keras.preprocessing import *\n\nn = 5\nfig, axes = plt.subplots(1, n, figsize=(25,10))\n\nfor i in range(n):\n    random_artist = random.choice(artists_top_name)\n    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))\n    random_image_file = os.path.join(images_dir, random_artist, random_image)\n\n    # Original image\n\n    test_image = image.load_img(random_image_file, target_size=(train_input_shape[0:2]))\n\n    # Predict artist\n    test_image = image.img_to_array(test_image)\n    test_image /= 255.\n    test_image = np.expand_dims(test_image, axis=0)\n\n    prediction = model.predict(test_image)\n    prediction_probability = np.amax(prediction)\n    prediction_idx = np.argmax(prediction)\n\n    labels = train_generator.class_indices\n    labels = dict((v,k) for k,v in labels.items())\n\n    #print(\"Actual artist =\", random_artist.replace('_', ' '))\n    #print(\"Predicted artist =\", labels[prediction_idx].replace('_', ' '))\n    #print(\"Prediction probability =\", prediction_probability*100, \"%\")\n\n    title = \"Actual artist = {}\\nPredicted artist = {}\\nPrediction probability = {:.2f} %\" \\\n                .format(random_artist.replace('_', ' '), labels[prediction_idx].replace('_', ' '),\n                        prediction_probability*100)\n\n    # Print image\n    axes[i].imshow(plt.imread(random_image_file))\n    axes[i].set_title(title)\n    axes[i].axis('off')\n\nplt.show()\n\n# %% [code] {\"_kg_hide-input\":false}\n# Predict from web - this is an image of Titian.\nurl = 'https://www.gpsmycity.com/img/gd/2081.jpg'\n\nimport imageio\nimport cv2\n\nweb_image = imageio.imread(url)\nweb_image = cv2.resize(web_image, dsize=train_input_shape[0:2], )\nweb_image = image.img_to_array(web_image)\nweb_image /= 255.\nweb_image = np.expand_dims(web_image, axis=0)\n\n\nprediction = model.predict(web_image)\nprediction_probability = np.amax(prediction)\nprediction_idx = np.argmax(prediction)\n\nprint(\"Predicted artist =\", labels[prediction_idx].replace('_', ' '))\nprint(\"Prediction probability =\", prediction_probability*100, \"%\")\n\nplt.imshow(imageio.imread(url))\nplt.axis('off')\nplt.show()\n\n# %% [code]\nprint(model.summary())\n\n# %% [code]\n","metadata":{"_uuid":"a6242167-b011-4949-b83a-62ac2b3e4f7e","_cell_guid":"6013d837-cec4-4b32-9a87-3299f81442d2","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}