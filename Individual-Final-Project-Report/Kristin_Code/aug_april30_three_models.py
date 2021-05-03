
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

import os
import cv2
#DataFolders
data_folder = "/content/drive/My Drive/Colab Notebooks/Xrays"
DATA_DIR = data_folder + "/TestData_Sertan"

data_COVID = DATA_DIR + "/COVID/"
data_NORMAL = DATA_DIR + "/Normal/"
data_VIRAL = DATA_DIR + "/Viral Pneumonia/"
data_LUNG = DATA_DIR + "/Lung_Opacity/"

RESIZE_TO = 299

x, y = [], []
for path in [f for f in os.listdir(data_COVID) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(data_COVID + path), (RESIZE_TO, RESIZE_TO)))
    y.append("COVID")
for path in [f for f in os.listdir(data_NORMAL) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(data_NORMAL + path), (RESIZE_TO, RESIZE_TO)))
    y.append("Normal")
for path in [f for f in os.listdir(data_VIRAL) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(data_VIRAL + path), (RESIZE_TO, RESIZE_TO)))
    y.append("Viral")
for path in [f for f in os.listdir(data_LUNG) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(data_LUNG + path), (RESIZE_TO, RESIZE_TO)))
    y.append("Lung")

print(' Number of Viral data:', len(os.listdir(data_VIRAL)))
print(' Number of Covid data:', len(os.listdir(data_COVID)))
print(' Number of Normal data:', len(os.listdir(data_NORMAL)))
print(' Number of Lung data:', len(os.listdir(data_LUNG)))

#y_test
from sklearn.preprocessing import LabelEncoder
y = np.array(y)
le = LabelEncoder()
le.fit(["COVID", "Normal", "Viral", "Lung"])
y = le.transform(y)

#Xception
x_cept = np.array(x)
x_cept = tf.keras.applications.xception.preprocess_input(x_cept, data_format = None)

RESIZE_TO = 224

x_res = []
for path in [f for f in os.listdir(data_COVID) if f[-4:] == ".png"]:
    x_res.append(cv2.resize(cv2.imread(data_COVID + path), (RESIZE_TO, RESIZE_TO)))
for path in [f for f in os.listdir(data_NORMAL) if f[-4:] == ".png"]:
    x_res.append(cv2.resize(cv2.imread(data_NORMAL + path), (RESIZE_TO, RESIZE_TO)))
for path in [f for f in os.listdir(data_VIRAL) if f[-4:] == ".png"]:
    x_res.append(cv2.resize(cv2.imread(data_VIRAL + path), (RESIZE_TO, RESIZE_TO)))
for path in [f for f in os.listdir(data_LUNG) if f[-4:] == ".png"]:
    x_res.append(cv2.resize(cv2.imread(data_LUNG + path), (RESIZE_TO, RESIZE_TO)))

#Resnet
x_res = np.array(x_res)
x_res = tf.keras.applications.resnet.preprocess_input(x_res, data_format = None)

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
model_1.load_weights(filepath=abspath_curr + '/result/model_1_aug.h5')

# Compile the model
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',    
              metrics=['accuracy'])

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
class_names_tr = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

y_pred_res = model_1.predict(x_res)
pred_res = np.argmax(y_pred_res, axis=1)

print(classification_report(y, pred_res, target_names=class_names_tr))
print("Accuracy : ", accuracy_score(y, pred_res) * 100)
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
model_2.load_weights(filepath=abspath_curr + '/result/model_2_aug.h5')

# Compile the model
model_2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
class_names_tr = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

y_pred_xcept = model_2.predict(x_cept)
pred_xcept = np.argmax(y_pred_xcept, axis=1)
print(classification_report(y, pred_xcept, target_names=class_names_tr))
print("Accuracy : ", accuracy_score(y, pred_xcept) * 100)
print("\n")

#Loading VGG16 Model

# Add the pretrained layers
pretrained_model = keras.applications.VGG16(include_top=False, weights='imagenet')

# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

# Add Dropout
drop = keras.layers.Dropout(0.4)
dropout = drop(average_pooling)

# Add the output layer
output = keras.layers.Dense(4, activation='softmax')(dropout)
model_3 = keras.Model(inputs=pretrained_model.input, outputs=output)

# Load the saved model
model_3.load_weights(filepath=abspath_curr + '/result/modelVGG16.h5')

# Compile the model
model_3.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Classification report - VGG16
y_pred_VGG16 = model_3.predict(x_res)
pred_VGG16 = np.argmax(y_pred_VGG16, axis=1)
print(classification_report(y, pred_VGG16, target_names=class_names_tr))
print('VGG16 Results')
print("Accuracy : ", accuracy_score(y, pred_VGG16) * 100)
print("\n")

import pandas as pd
#Combining the models
df1 = pd.DataFrame(y_pred_res, columns = ['COVID1', 'Lung_Opacity1', 'Normal1', 'Viral Pneumonia1'])
df2 = pd.DataFrame(y_pred_xcept, columns = ['COVID2', 'Lung_Opacity2', 'Normal2', 'Viral Pneumonia2'])
df3 = pd.DataFrame(y_pred_VGG16, columns = ['COVID3', 'Lung_Opacity3', 'Normal3', 'Viral Pneumonia3'])
df_target = pd.DataFrame(y, columns = ['target'])

df1['COVID2'] = df2['COVID2']
df1['Lung_Opacity2'] = df2['Lung_Opacity2']
df1['Normal2'] = df2['Normal2']
df1['Viral Pneumonia2'] = df2['Viral Pneumonia2']

df1['COVID3'] = df3['COVID3']
df1['Lung_Opacity3'] = df3['Lung_Opacity3']
df1['Normal3'] = df3['Normal3']
df1['Viral Pneumonia3'] = df3['Viral Pneumonia3']

df1['target'] = df_target['target']



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

#clean the dataset
print("Sum of NULL values in each column. ")
print(data.isnull().sum())

#split the dataset
# separate the predictor and target variable
X = data.values[:, 0:12]
Y = data.values[:, 12]

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

#perform training with random forest with all columns
# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)

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



