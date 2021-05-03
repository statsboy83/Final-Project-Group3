
# 1) Introduction of the Dataset

Sertan Akinci
Kristin Levine

We decided to work on this Kaggle dataset:
https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

# 2) Setup -- Creating a Test Set
"""


import sys
import os
import random
import numpy as np



"""Setting the Paths"""

# Get the absolute path of the current folder
abspath_curr = '/content/drive/My Drive/Colab Notebooks/Xrays'

"""**Importing Libraries**"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline 

# Set matplotlib sizes
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=20)

# Commented out IPython magic to ensure Python compatibility.
# The magic below allows us to use tensorflow version 2.x
# %tensorflow_version 2.x 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
#import tensorflow_datasets as tfds
import numpy as np

"""# 3) Loading the Data -- ResNet50

Figuring out how to load the data into the pipeline we've used in class was one of the most challenging parts of this competition. Since the pipeline we've used in class uses Tensorflow databases, it wasn't immediately obvious to us how to load this new data.  (https://github.com/yuxiaohuang/teaching/blob/master/gwu/machine_learning_I/fall_2020/code/p3_deep_learning/p3_c2_supervised_learning/p3_c2_s3_convolutional_neural_networks/case_study/case_study.ipynb) 

We started by looking at the other notebooks people had used for this competition, but ultimately uncovered a more elegant solution:
using tf.keras.preprocessing.image_dataset_from_directory() 

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

**(a) Setting Parameters**

Note: We tried a bunch of different parameters for ResNet -- these were the ones that ended up working best. See "conclusions" for a complete listing of all parameters tried.
"""

img_size=[224,224]
batch = 32
split = 0.2

epoch_n = 5
random_seed = 42

# Set random seed in tensorflow
tf.random.set_seed(random_seed)

# Set random seed in numpy
np.random.seed(random_seed)

#https://github.com/yuxiaohuang/teaching/blob/master/gwu/machine_learning_I/fall_2020/code/utilities/p3_deep_learning/pmlm_utilities_deep.ipynb

def preprocess_pretrain(data, label):
    """
    Preprocess the data using pretrained model

    Parameters
    ----------
    data: the data
    label: the label
    
    Returns
    ----------
    The preprocessed data using pretrained model
    """

    # Preprocess the data
    data_preprocessed = preprocess_input(data)

    return data_preprocessed, label

"""**(b) Loading the Data**"""

#DataFolders
data_folder = "/content/drive/My Drive/Colab Notebooks/Xrays/COVID-19_Radiography_Dataset"
train_data = data_folder + "/TrainData/"
test_data = data_folder + "/TestData/"

data_tr = tf.keras.preprocessing.image_dataset_from_directory(
    train_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=split, subset='training', interpolation='bilinear'
)

data_val = tf.keras.preprocessing.image_dataset_from_directory(
    train_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=split, subset='validation', interpolation='bilinear'
)

data_te = tf.keras.preprocessing.image_dataset_from_directory(
    test_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=None, subset=None, interpolation='bilinear'
)

"""**(c) Quick Validations**"""

#Training Data
class_names_tr = data_tr.class_names
print(class_names_tr)

#for images, labels in data_tr.take(1): 
#  print(images.shape, labels.shape)

#Validation Data
class_names_val = data_val.class_names
print(class_names_val)

#for images, labels in data_val.take(1): 
#  print(images.shape, labels.shape)

#Test Data
class_names_te = data_te.class_names
print(class_names_te)

#for images, labels in data_te.take(1): 
#  print(images.shape, labels.shape)

"""**(d) Viewing the Images**"""

plt.figure(figsize=(10, 10))
for images, labels in data_te.take(1):
  for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names_te[labels[i]])
    plt.axis("off")

"""#4) Using Class Pipeline - ResNet50

After loading the data as a tensorflow dataset, we were able to use the pipeline we learned in class.

(a) Preprocessing the Data
"""

# Set the preprocess_input of the pretrained model
global preprocess_input
preprocess_input = tf.keras.applications.resnet.preprocess_input

# Preprocess the training data using pretrained model
data_train = data_tr.map(preprocess_pretrain)

# Preprocess the validation data using pretrained model
data_valid = data_val.map(preprocess_pretrain)

# Preprocess the test data using pretrained model
data_test = data_te.map(preprocess_pretrain)

"""(b) Creating the Model"""

# Make directory
directory = os.path.dirname(abspath_curr + '/result/')
if not os.path.exists(directory):
    os.makedirs(directory)

"""Note: We ended up adding a dropout layer to our model -- we tried different dropout amounts; this one was the value that worked best.  See conclusion for details."""

# Add the pretrained layers
pretrained_model = keras.applications.ResNet50(include_top=False, weights='imagenet')

# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

# Add Dropout 
drop = keras.layers.Dropout(0.2)
dropout = drop(average_pooling)

# Add the output layer
output = keras.layers.Dense(len(class_names_tr), activation='softmax')(dropout)

# Get the model
model = keras.Model(inputs=pretrained_model.input, outputs=output)

model.summary()

"""**(c) Freeze and Train**"""

# For each layer in the pretrained model
for layer in pretrained_model.layers:
    # Freeze the layer
    layer.trainable = False

# ModelCheckpoint callback
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=abspath_curr + '/result/model_1.h5',
                                                      save_best_only=True,
                                                      save_weights_only=True)

# EarlyStopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

# ReduceLROnPlateau callback
reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                            patience=1)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',    
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model.fit(data_train,
                    epochs=5,
                    validation_data=data_valid,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])

"""**(d) Plotting the Learning Curve**"""

# Make directory
directory = os.path.dirname(abspath_curr + '/result/figure_1/')
if not os.path.exists(directory):
    os.makedirs(directory)

import pandas as pd

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Set grid
plt.grid(True)

# Save and show the figure
plt.tight_layout()
plt.savefig(abspath_curr + '/result/figure_1/learning_curve_before_unfreezing.pdf')
plt.show()

"""**(e) Unfreeze the Pretrained Layers and Train again**"""

# For each layer in the pretrained model
for layer in pretrained_model.layers:
    # Unfreeze the layer
    layer.trainable = True

"""Note: We also tried different learning rates -- see conclusion for details."""

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model.fit(data_train,
                    epochs=5,
                    validation_data=data_valid,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])

"""**(f) Plotting the Learning Curve and Evaluating the Model**"""

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Set grid
plt.grid(True)

# Save and show the figure
plt.tight_layout()
plt.savefig(abspath_curr + '/result/figure_1/learning_curve_after_unfreezing.pdf')
plt.show()

# Load the saved model
model.load_weights(filepath=abspath_curr + '/result/model_1.h5')

loss, accuracy = model_1.evaluate(data_test)

y_pred = model.predict(data_test)
print(y_pred.round())
print(len(y_pred))

"""# 5) Loading the Data -- Xception

Next we wanted to see if we could use another pretrained model with this data. 

Note: this model takes a different image size, and we selected slightly different parameters for the best model, so we had to reload the data.

**(a) Setting Parameters**
"""

img_size=[299,299]
batch = 32
split = 0.2
epoch_n = 5
random_seed = 42

# Set random seed in tensorflow
tf.random.set_seed(random_seed)

# Set random seed in numpy
np.random.seed(random_seed)



"""**(b) Loading the Data**"""

data_tr = tf.keras.preprocessing.image_dataset_from_directory(
    train_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=split, subset='training', interpolation='bilinear'
)

data_val = tf.keras.preprocessing.image_dataset_from_directory(
    train_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=split, subset='validation', interpolation='bilinear'
)

data_te = tf.keras.preprocessing.image_dataset_from_directory(
    test_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=None, subset=None, interpolation='bilinear'
)

"""**(c) Quick Validations**"""

#Training Data
class_names_tr = data_tr.class_names
print(class_names_tr)

#for images, labels in data_tr.take(1): 
#  print(images.shape, labels.shape)

#Validation Data
class_names_val = data_val.class_names
print(class_names_val)

#for images, labels in data_val.take(1): 
#  print(images.shape, labels.shape)

#Test Data
class_names_te = data_te.class_names
print(class_names_te)

#for images, labels in data_te.take(1): 
#  print(images.shape, labels.shape)

"""**(d) Viewing the Images**"""

plt.figure(figsize=(10, 10))
for images, labels in data_te.take(1):
  for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names_te[labels[i]])
    plt.axis("off")

"""#6) Using Class Pipeline - Xception

**(a) Preprocessing the Data**
"""

# Set the preprocess_input of the pretrained model
global preprocess_input

preprocess_input = tf.keras.applications.xception.preprocess_input

# Preprocess the training data using pretrained model
data_train = data_tr.map(preprocess_pretrain)

# Preprocess the validation data using pretrained model
data_valid = data_val.map(preprocess_pretrain)

# Preprocess the test data using pretrained model
data_test = data_te.map(preprocess_pretrain)

"""**(b) Creating the Model**"""

# Make directory
directory = os.path.dirname(abspath_curr + '/result/')
if not os.path.exists(directory):
    os.makedirs(directory)

"""Note: This time we used a 0.5 dropout layer."""

# Add the pretrained layers
pretrained_model = keras.applications.Xception(include_top=False, weights='imagenet')

# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

# Add Dropout 50%
drop = keras.layers.Dropout(0.5)
dropout = drop(average_pooling)

# Add the output layer
output = keras.layers.Dense(len(class_names_tr), activation='softmax')(dropout)

# Get the model
model = keras.Model(inputs=pretrained_model.input, outputs=output)

model.summary()

"""**(c) Freeze and Train**"""

# For each layer in the pretrained model
for layer in pretrained_model.layers:
    # Freeze the layer
    layer.trainable = False

# ModelCheckpoint callback
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=abspath_curr + '/result/model_2.h5',
                                                      save_best_only=True,
                                                      save_weights_only=False)

# EarlyStopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

# ReduceLROnPlateau callback
reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                            patience=2)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model.fit(data_train,
                    epochs=epoch_n,
                    validation_data=data_valid,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])

# Save the model seperately incase you wanna come back to this point
model.save(abspath_curr + '/result/model_2/model_2_b4freez.h5')

"""**(d) Plotting the Learning Curve**"""

# Make directory
directory = os.path.dirname(abspath_curr + '/result/figure_2/')
if not os.path.exists(directory):
    os.makedirs(directory)

import pandas as pd

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Set grid
plt.grid(True)

# Save and show the figure
plt.tight_layout()
plt.savefig(abspath_curr + '/result/figure_2/learning_curve_before_unfreezing.pdf')
plt.show()

"""**(e) Unfreeze the Pretrained Layers and Train again**"""

# For each layer in the pretrained model
for layer in pretrained_model.layers:
    # Unfreeze the layer
    layer.trainable = True

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model.fit(data_train,
                    epochs=epoch_n,
                    validation_data=data_valid,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])

"""**(f) Plotting the Learning Curve and Evaluating the Model**"""

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Set grid
plt.grid(True)

# Save and show the figure
plt.tight_layout()
plt.savefig(abspath_curr + '/result/figure_2/learning_curve_after_unfreezing.pdf')
plt.show()

# Load the saved model
model_2 = model.load_weights(filepath=abspath_curr + '/result/model_2.h5')

loss, accuracy = model_2.evaluate(data_test)



# load models from file



#https://towardsdatascience.com/destroy-image-classification-by-ensemble-of-pre-trained-models-f287513b7687


#y_pred = model.predict(data_test)
print(y_pred.round())
print(len(y_pred))

y_pred2 = model.predict(data_test)
print(y_pred2.round())
print(len(y_pred2))

y_combined = (y_pred + y_pred2)/2
print(y_combined.round())

"""# 7) Conclusions

ADD CONCLUSIONS HERE/ADD PARAMETERS FOR NEW PROJECT
"""





