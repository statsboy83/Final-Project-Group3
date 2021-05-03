
#####################################################################
# DESCRIPTION: This program is used to allocate a 10% Test data     #
#              as well as augment and over/under sample minority and#
#              majority classes in the training data respectively    #
#####################################################################


import numpy as np
import os
import matplotlib.pyplot as plt
import random
import cv2

random.seed(13)     # so that our Test Data is the same
test_pct = 0.1      # set this equal to the percentage you want your test data to be. Default 10%.

# Location of the data files
data_folder = "FinalProject/COVID-19_Radiography_Dataset/TrainData"
covid_data = data_folder + "/COVID/"
lung_data = data_folder + "/Lung_Opacity/"
norm_data = data_folder + "/Normal/"
viral_data = data_folder + "/Viral Pneumonia/"

#Test
test_data = "FinalProject/COVID-19_Radiography_Dataset/TestData"
covid_test = test_data + "/COVID/"
lung_test = test_data + "/Lung_Opacity/"
norm_test = test_data + "/Normal/"
viral_test = test_data + "/Viral Pneumonia/"

print(' Number of Viral data:', len(os.listdir(viral_data)))
print(' Number of Covid data:', len(os.listdir(covid_data)))
print(' Number of Normal data:', len(os.listdir(norm_data)))
print(' Number of Lung data:', len(os.listdir(lung_data)))

# Drawing a random sample of 10% of the data in each folder
cov_l = random.sample(os.listdir(covid_data), np.int(np.floor(len(os.listdir(covid_data)) * test_pct)))
lung_l = random.sample(os.listdir(lung_data), np.int(np.floor(len(os.listdir(lung_data)) * test_pct)))
nom_l = random.sample(os.listdir(norm_data), np.int(np.floor(len(os.listdir(norm_data)) * test_pct)))
viral_l = random.sample(os.listdir(viral_data), np.int(np.floor(len(os.listdir(viral_data)) * test_pct)))

#Moving the files from their respective folder to TestData folder
for l in cov_l:
    os.rename(covid_data + l, covid_test + l)
for l in lung_l:
    os.rename(lung_data + l, lung_test + l)
for l in nom_l:
    os.rename(norm_data + l, norm_test + l)
for l in viral_l:
    os.rename(viral_data + l, viral_test + l)

print(' Number of Viral data:', len(os.listdir(viral_data)))
print(' Number of Covid data:', len(os.listdir(covid_data)))
print(' Number of Normal data:', len(os.listdir(norm_data)))
print(' Number of Lung data:', len(os.listdir(lung_data)))


########## Augmenting and Oversampling Undersampling Images #####

# Oversample Viral and COVID classes
AUG_DIR = 'FinalProject/COVID-19_Radiography_Dataset/Augs/'
listc = os.listdir(covid_data)
for i in range(0,3258,8):
    img = cv2.imread(covid_data + listc[i])
    cv2.imwrite(covid_data + 'augfh' + str(i) + '.png', cv2.flip(img, 1))
    cv2.imwrite(covid_data + 'augfv' + str(i) + '.png', cv2.flip(img, 0))
    cv2.imwrite(covid_data + 'augrt' + str(i) + '.png', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite(covid_data + 'augrt' + str(i) + '.png', cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

listv = os.listdir(viral_data)
for i in range(len(listv)):
    img = cv2.imread(viral_data + listv[i])
    cv2.imwrite(viral_data + 'augfh' + str(i) + '.png', cv2.flip(img, 1))
    cv2.imwrite(viral_data + 'augfv' + str(i) + '.png', cv2.flip(img, 0))
    cv2.imwrite(viral_data + 'augrt' + str(i) + '.png', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite(viral_data + 'augrt' + str(i) + '.png', cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    
pic = cv2.imread(AUG_DIR + 'augrn0.png')
plt.imshow(pic)
plt.show()

# Undersampling NORMAL Class #####
nom_l = random.sample(os.listdir(norm_data), np.int(np.floor(len(os.listdir(norm_data)) * 0.44)))
for l in nom_l:
    os.rename(norm_data + l, AUG_DIR + l)
