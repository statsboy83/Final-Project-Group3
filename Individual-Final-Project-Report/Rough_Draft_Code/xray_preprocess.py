

import sys
import os



# Get the absolute path of the current folder
#abspath_curr = '/content/drive/My Drive/Colab Notebooks/Xrays'

import random
import sys
import os
import numpy as np
random.seed(13)     # so that our Test Data is the same
test_pct = 0.1      # set this equal to the percentage you want your test data to be. Default 10%

# Location of the data files
#Train
data_folder = "/content/drive/My Drive/Colab Notebooks/Xrays/Whole_dataset/COVID-19_Radiography_Dataset/TrainData"
covid_data = data_folder + "/COVID/"
lung_data = data_folder + "/Lung_Opacity/"
norm_data = data_folder + "/Normal/"
viral_data = data_folder + "/Viral_Pneumonia/"

#Test
test_data = "/content/drive/My Drive/Colab Notebooks/Xrays/Whole_dataset/COVID-19_Radiography_Dataset/TestData"
covid_test = test_data + "/COVID/"
lung_test = test_data + "/Lung_Opacity/"
norm_test = test_data + "/Normal/"
viral_test = test_data + "/Viral_Pneumonia/"

# Drawing a random sample of 10% of the data in each folder
cov_l = random.sample(os.listdir(covid_data), np.int(np.floor(len(os.listdir(covid_data)) * test_pct)))
#lung_l = random.sample(os.listdir(lung_data), np.int(np.floor(len(os.listdir(lung_data)) * test_pct)))
#nom_l = random.sample(os.listdir(norm_data), np.int(np.floor(len(os.listdir(norm_data)) * test_pct)))
#viral_l = random.sample(os.listdir(viral_data), np.int(np.floor(len(os.listdir(viral_data)) * test_pct)))

#Moving the files from their respective folder to TestData/respective folder
for l in cov_l:
    os.rename(covid_data + l, covid_test + l)
#for l in lung_l:
#    os.rename(lung_data + l, lung_test + l)
#for l in nom_l:
#    os.rename(norm_data + l, norm_test + l)
#for l in viral_l:
#    os.rename(viral_data + l, viral_test + l)

