
#############################################################
# DESCRIPTION: This program trains our model using Vgg16    #
#              pre-trained model.                           #
#############################################################

import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# FOLDER ASSIGNMENT
abspath_curr = "COVID-19_Radiography_Dataset/"
data_folder = "COVID-19_Radiography_Dataset/TrainData/"
covid_data = data_folder + "COVID/"
lung_data = data_folder + "Lung_Opacity/"
norm_data = data_folder + "Normal/"
viral_data = data_folder + "Viral Pneumonia/"

#Test
test_data = "COVID-19_Radiography_Dataset/TestData/"
covid_test = test_data + "COVID/"
lung_test = test_data + "Lung_Opacity/"
norm_test = test_data + "Normal/"
viral_test = test_data + "Viral Pneumonia/"

print(' Number of Viral data:', len(os.listdir(viral_data)))
print(' Number of Covid data:', len(os.listdir(covid_data)))
print(' Number of Normal data:', len(os.listdir(norm_data)))
print(' Number of Lung data:', len(os.listdir(lung_data)))


# PARAMETERS
batch = 32
split = 0.2
img_size = [244, 244]
random_seed = 13
n_epoch = 20
random.seed(13)     # so that our Test Data is the same
test_pct = 0.1      # set this equal to the percentage you want your test data to be. Default 10%.

# DATA LOAD
data_tr = tf.keras.preprocessing.image_dataset_from_directory(
    data_folder, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=split, subset='training', interpolation='bilinear')

data_val = tf.keras.preprocessing.image_dataset_from_directory(
    data_folder, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size=batch, image_size=img_size, seed=random_seed,
    validation_split=split, subset='validation', interpolation='bilinear')


class_names_tr = data_tr.class_names
print(class_names_tr)
for images, labels in data_tr.take(1):
  print(images.shape, labels.shape)


# DATA PROCESS
preprocess_input = tf.keras.applications.vgg16.preprocess_input
def preprocess_pretrain(data, label):
    data_preprocessed = preprocess_input(data)
    return data_preprocessed, label

# Preprocess the training data using pretrained model
data_train = data_tr.map(preprocess_pretrain)
data_valid = data_val.map(preprocess_pretrain)


# Add the pretrained layers
pretrained_model = keras.applications.VGG16(include_top=False, weights='imagenet')
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)
drop = keras.layers.Dropout(0.4)
dropout = drop(average_pooling)
output = keras.layers.Dense(len(class_names_tr), activation='softmax')(dropout)
model = keras.Model(inputs=pretrained_model.input, outputs=output)
model.summary()


# Freeze all the layers
for layer in pretrained_model.layers:
    layer.trainable = False



# ModelCheckpoint callback
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=abspath_curr + 'Output/modelVGG16.h5',
                                                      save_best_only=True,
                                                      save_weights_only=True)
# EarlyStopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)
# ReduceLROnPlateau callback
reduce_lr_on_plateau_cb = keras.callbacks.ReduceLROnPlateau(factor=0.1,
                                                            patience=2)


# Compile the model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train, evaluate and save the best model
history = model.fit(data_train,
                    epochs=n_epoch,
                    validation_data=data_valid,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])


# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.tight_layout()
plt.savefig(abspath_curr + 'Output/learning_curve_freezing.pdf')
plt.show()


# Unfreeze pretrained and train again
for layer in pretrained_model.layers:
    layer.trainable = True


# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train, evaluate and save the best model
history = model.fit(data_train,
                    epochs=n_epoch,
                    validation_data=data_valid,
                    callbacks=[model_checkpoint_cb,
                               early_stopping_cb,
                               reduce_lr_on_plateau_cb])


# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.tight_layout()
plt.savefig(abspath_curr + 'Output/learning_curve_after_unfreezing.pdf')
plt.show()


# Reading Test Data
RESIZE_TO = 244

x, y = [], []
for path in [f for f in os.listdir(covid_test) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(covid_test + path), (RESIZE_TO, RESIZE_TO)))
    y.append("COVID")
for path in [f for f in os.listdir(norm_test) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(norm_test + path), (RESIZE_TO, RESIZE_TO)))
    y.append("Normal")
for path in [f for f in os.listdir(viral_test) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(viral_test + path), (RESIZE_TO, RESIZE_TO)))
    y.append("Viral")
for path in [f for f in os.listdir(lung_test) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(lung_test + path), (RESIZE_TO, RESIZE_TO)))
    y.append("Lung")



# y_test
class_names_tr = ["COVID", "Normal", "Viral", "Lung"]
y = np.array(y)
le = LabelEncoder()
le.fit(class_names_tr)
yt = le.transform(y)

# x_test
x = np.array(x)
x_test = preprocess_input(x, data_format = None)

y_pred = model.predict(x_test)
pred = np.argmax(y_pred, axis=1)

print(classification_report(yt, pred, target_names=class_names_tr))
print("Accuracy : ", accuracy_score(yt, pred) * 100)
print("\n")