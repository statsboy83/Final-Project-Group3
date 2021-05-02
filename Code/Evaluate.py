##################################################################
# DESCRIPTION: This program uploads all our trained models,      #
#              make predictions, and combines them to run through#
#              RANDOM FOREST                                     #
##################################################################

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# FOLDER ASSIGNMENT
class_names_tr = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
abspath_curr = 'COVID-19_Radiography_Dataset/'
test_data = "COVID-19_Radiography_Dataset/TestData/"
covid_test = test_data + "COVID/"
lung_test = test_data + "Lung_Opacity/"
norm_test = test_data + "Normal/"
viral_test = test_data + "Viral Pneumonia/"

print(' Number of Viral data:', len(os.listdir(viral_test)))
print(' Number of Covid data:', len(os.listdir(covid_test)))
print(' Number of Normal data:', len(os.listdir(norm_test)))
print(' Number of Lung data:', len(os.listdir(lung_test)))


################# FUNCTIONS #######################

# FUNCTION TO LOAD DATA
def load_testdata(model_name, RESIZE_TO):
    x, y = [], []
    for path in os.listdir(covid_test):
        x.append(cv2.resize(cv2.imread(covid_test + path), (RESIZE_TO, RESIZE_TO)))
        y.append("COVID")
    for path in os.listdir(norm_test):
        x.append(cv2.resize(cv2.imread(norm_test + path), (RESIZE_TO, RESIZE_TO)))
        y.append("Normal")
    for path in os.listdir(viral_test):
        x.append(cv2.resize(cv2.imread(viral_test + path), (RESIZE_TO, RESIZE_TO)))
        y.append("Viral")
    for path in os.listdir(lung_test):
        x.append(cv2.resize(cv2.imread(lung_test + path), (RESIZE_TO, RESIZE_TO)))
        y.append("Lung")

    y = np.array(y)
    le = LabelEncoder()
    le.fit(["COVID", "Normal", "Viral", "Lung"])
    y = le.transform(y)

    x = np.array(x)
    if model_name == 'xception':
        x = tf.keras.applications.xception.preprocess_input(x, data_format=None)
        print('Xception is chosen')
    elif model_name == 'resnet':
        x = tf.keras.applications.resnet50.preprocess_input(x, data_format=None)
        print('RESNET is chosen')
    elif model_name == 'vgg16':
        x = tf.keras.applications.vgg16.preprocess_input(x, data_format=None)
        print('VGG16 is chosen')
    return x, y


# FUNCTION TO LOAD MODEL
def load_model(model_name):
    if model_name == 'xception':
        pretrained_model = keras.applications.Xception(include_top=False, weights='imagenet')
        model_location = abspath_curr + 'Output/modelXception.h5'
        print('Xception is chosen')
    elif model_name == 'resnet':
        pretrained_model = keras.applications.ResNet50(include_top=False, weights='imagenet')
        model_location = abspath_curr + 'Output/modelRESNET.h5'
        print('RESNET is chosen')
    elif model_name == 'vgg16':
        pretrained_model = keras.applications.VGG16(include_top=False, weights='imagenet')
        model_location = abspath_curr + 'Output/modelVGG16.h5'
        print('VGG16 is chosen')

    # The model
    average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)
    drop = keras.layers.Dropout(0.2)
    dropout = drop(average_pooling)
    output = keras.layers.Dense(4, activation='softmax')(dropout)
    model = keras.Model(inputs=pretrained_model.input, outputs=output)

    model.load_weights(filepath=model_location)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


# FUNCTION FOR CLASSIFICATION REPORT
def class_report(x, y, model):
    y_pred = model.predict(x)
    pred = np.argmax(y_pred, axis=1)
    print(classification_report(y, pred, target_names=class_names_tr))
    print("Accuracy : ", accuracy_score(y, pred) * 100)
    print("\n")
    return y_pred


################# PROGRAM BEGINS #######################

# LOADING EACH MODEL AND SHOWING INDIVIDUAL PERFORMANCES
x_test, y_test = load_testdata('resnet', 244)
mymodel = load_model('resnet')
p1 = class_report(x_test, y_test, mymodel)

x_test, y_test = load_testdata('xception', 299)
mymodel = load_model('xception')
p2 = class_report(x_test, y_test, mymodel)

x_test, y_test = load_testdata('vgg16', 244)
mymodel = load_model('vgg16')
p3 = class_report(x_test, y_test, mymodel)


# COMBINING THE MODEL PREDICTIONS AND RUNNING RANDOM FOREST (ENSEMBLE MODEL)
X = np.concatenate((p1, p2, p3), axis=1)
Y = y_test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

print("\n")
print("Results Using All Features: \n")
print("Classification Report: ")
print(classification_report(y_test, y_pred))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

conf_matrix = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(conf_matrix, index=class_names_tr, columns=class_names_tr)
plt.figure(figsize=(15, 15))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.tight_layout()
plt.show()
