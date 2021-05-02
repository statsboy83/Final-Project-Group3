Augment.py : This program is used first after downloading the data to allocate a 10% Test dataset as well as augment and over/under sample minority and majority 
             classes in the traning data respectively.

ResNet.py  : This program was run after the data augmentation. It trains and saves the model with REsNet50 pre-trained layers.
VGG16.py   : This program was run after the data augmentation. It trains and saves the model with VGG16 pre-trained layers.
Xception.py: This program was run after the data augmentation. It trains and saves the model with Xception pre-trained layers.

Evaluate.py: This program uploads all our trained models, make predictions and combines them to runthrough Random Forest for ensamble. 
