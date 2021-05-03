
import sys
import os
import random
import numpy as np
import cv2



# Commented out IPython magic to ensure Python compatibility.
# The magic below allows us to use tensorflow version 2.x
# %tensorflow_version 2.x 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

# Get the absolute path of the current folder
abspath_curr = '/content/drive/My Drive/Colab Notebooks/Xrays'
batch = 32
random_seed = 42

# Set random seed in tensorflow
tf.random.set_seed(random_seed)

# Set random seed in numpy
np.random.seed(random_seed)

"""#Loading Data"""

#DataFolders
data_folder = "/content/drive/My Drive/Colab Notebooks/Xrays/COVID-19_Radiography_Dataset"
test_data = data_folder + "/TestData"

test_data_resnet = tf.keras.preprocessing.image_dataset_from_directory(
    test_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size = 60, image_size=[224, 224], seed=random_seed,
    validation_split=None, subset=None, interpolation='bilinear'
)

test_data_xcept = tf.keras.preprocessing.image_dataset_from_directory(
    test_data, labels='inferred',  class_names=None,
    color_mode='rgb', batch_size = 60, image_size=[299, 299], seed=random_seed,
    validation_split=None, subset=None, interpolation='bilinear'
)

y_xcep = np.concatenate([y for x, y in test_data_resnet], axis=0)
y_res = np.concatenate([y for x, y in test_data_xcept], axis=0)
print(y_xcep)
print(y_res)

if (y_xcep[0] == y_res[0]): 
  print("Same")
else:
  print("Not same")

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

# Set the preprocess_input of the pretrained model -- Resnet
global preprocess_input
preprocess_input = tf.keras.applications.resnet.preprocess_input

# Preprocess the test data using pretrained model
test_data_resnet = test_data_resnet.map(preprocess_pretrain)

for images, labels in test_data_resnet.take(1):
  print(images.shape, labels.shape)

# Set the preprocess_input of the pretrained model -- Xception
global preprocess_input
preprocess_input = tf.keras.applications.xception.preprocess_input

# Preprocess the test data using pretrained model
test_data_xcept = test_data_xcept.map(preprocess_pretrain)

for images, labels in test_data_xcept.take(1):
  print(images.shape, labels.shape)

"""#Loading Resnet Model"""

#Loading Resnet Model

# Add the pretrained layers
pretrained_model = keras.applications.ResNet50(include_top=False, weights='imagenet')

# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

# Add Dropout 
drop = keras.layers.Dropout(0.2)
dropout = drop(average_pooling)

# Add the output layer
output = keras.layers.Dense(4, activation='softmax')(dropout)

# Get the model
model_1 = keras.Model(inputs=pretrained_model.input, outputs=output)

#model_1.summary()

#Load the saved model
model_1.load_weights(filepath=abspath_curr + '/result/model_1.h5')

# Compile the model
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',    
              metrics=['accuracy'])

#Resnet Predictions
loss, accuracy = model_1.evaluate(test_data_resnet)
y_pred_resnet = model_1.predict(test_data_resnet)
print(y_pred_resnet.round())
print(len(y_pred_resnet))

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
class_names_tr = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

y_pred = model_1.predict(test_data_resnet)
pred_indx = np.argmax(y_pred, axis=1)
y_test_label = np.concatenate([y for x, y in test_data_resnet], axis=0)
print(classification_report(y_test_label, pred_indx, target_names=class_names_tr))
print("Accuracy : ", accuracy_score(y_test_label, pred_indx) * 100)
print("\n")

"""#Loading Xception Model"""

#Loading Xception Model

# Add the pretrained layers
pretrained_model = keras.applications.Xception(include_top=False, weights='imagenet')

# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

# Add Dropout 50%
drop = keras.layers.Dropout(0.5)
dropout = drop(average_pooling)

# Add the output layer
output = keras.layers.Dense(4, activation='softmax')(dropout)

# Get the model
model_2 = keras.Model(inputs=pretrained_model.input, outputs=output)

#model_2.summary()

# Load the saved model
model_2.load_weights(filepath=abspath_curr + '/result/model_2.h5')

# Compile the model
model_2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Xception Predictions

loss, accuracy = model_2.evaluate(test_data_xcept)
y_pred_xcept = model_2.predict(test_data_xcept)
print(y_pred_xcept.round())
print(len(y_pred_xcept))





y_pred = model_2.predict(test_data_xcept)
pred_indx = np.argmax(y_pred, axis=1)
y_test_label = np.concatenate([y for x, y in test_data_xcept], axis=0)
print(classification_report(y_test_label, pred_indx, target_names=class_names_tr))
print("Accuracy : ", accuracy_score(y_test_label, pred_indx) * 100)
print("\n")

y_xcep = np.concatenate([y for x, y in test_data_resnet], axis=0)
y_res = np.concatenate([y for x, y in test_data_xcept], axis=0)
print(y_xcep)
print(y_res)

if (y_xcep[0] == y_res[0]): 
  print("Same")
else:
  print("Not same")



import pandas as pd
#print(y_pred_xcept)
df1 = pd.DataFrame(y_pred_resnet, columns = ['COVID1', 'Lung_Opacity1', 'Normal1', 'Viral Pneumonia1'])
df2 = pd.DataFrame(y_pred_xcept, columns = ['COVID2', 'Lung_Opacity2', 'Normal2', 'Viral Pneumonia2'])
#df_target = np.concatenate([y for x, y in test_data_xcept], axis=0)
df_target = pd.DataFrame(y_xcep, columns = ['target'])

df1['COVID2'] = df2['COVID2']
df1['Lung_Opacity2'] = df2['Lung_Opacity2']
df1['Normal2'] = df2['Normal2']
df1['Viral Pneumonia2'] = df2['Viral Pneumonia2']
df1['target'] = df_target['target']
print(df1)

#https://github.com/amir-jafari/Data-Mining/blob/master/9-Random%20Forest/1-Example_Exercise/RF_1.py
# %%%%%%%%%%%%% Random Forest  %%%%%%%%%%%%%%%%%%%%%%%%%%
#%%-----------------------------------------------------------------------
# Importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# import Dataset
# read data as panda dataframe
data = df1

# printing the dataswet rows and columns
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])

# printing the dataset obseravtions
print("Dataset first few rows:\n ")
print(data.head(2))

# printing the struture of the dataset
print("Dataset info:\n ")
print(data.info())

# printing the summary statistics of the dataset
print(data.describe(include='all'))

#clean the dataset
print("Sum of NULL values in each column. ")
print(data.isnull().sum())

#split the dataset
# separate the predictor and target variable
X = data.values[:, 0:8]
Y = data.values[:, 8]

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

#perform training with random forest with all columns
# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(X_train, y_train)

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# calculate metrics gini model

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

#print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()



