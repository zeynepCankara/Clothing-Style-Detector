# -*- coding: utf-8 -*-
"""
Verion: Fri Jun 20 2018

@author: Zeynep Cankara

Dataset manipulation and doing training/test split
"""

"********** Cleaning the dataset for training/testing *****************"
#import necessary libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.preprocessing import image
import pandas as pd
import numpy as np
import PIL 
from sklearn.model_selection import train_test_split
"""
    Preprocessing of the dataset
"""
# Importing the training set
dataset = pd.read_csv('data/clothes_labels.csv')

#get information about the dataset
dataset.info()
dataset.describe()

#get rid of the null data or label as empty
dataset.fillna('empty',inplace = True) #look info now everything looks normal

#make url correct format
dataset['url'].replace("https://s3.amazonaws.com/stylyze/", "", regex=True, inplace=True)

#get rid of unnecessary columns
dataset.drop("city_id",axis = 1, inplace = True) 
dataset.drop("multiple_layers",axis = 1, inplace = True) 
dataset.drop("created_time",axis = 1, inplace = True) 
dataset.drop("month_id",axis = 1, inplace = True) 
dataset.drop("lat",axis = 1, inplace = True) 
dataset.drop("long",axis = 1, inplace = True)  
dataset.drop("id",axis = 1, inplace = True) 

# Plot Category of clothes
fig_dims = (2,2)
plt.subplot2grid(fig_dims, (1, 1))
dataset['clothing_category'].value_counts().plot(kind='bar', 
                                    title='number of classes for clothes') #so you omly have 7 categories, drop the empty data
#get rid of empty data
d = dataset[dataset.clothing_category != 'empty']

#visualize the clothing categories
fig_dims = (2,2)
plt.subplot2grid(fig_dims, (1, 1))
d['clothing_category'].value_counts().plot(kind='bar', 
                                    title='number of classes for clothes') #cleanised
#rename the columns
d = d.rename(index = str, columns = {'url':'filename', 'x1': 'xmin', 'y1':'ymin', 'x2':'xmax', 'y2':'ymax', 'clothing_category':'classes'})

#keep dropping the column information
d.drop("wearing_necktie",axis = 1, inplace = True) 
d.drop("collar_presence",axis = 1, inplace = True) 
d.drop("wearing_scarf",axis = 1, inplace = True) 
d.drop("sleeve_length",axis = 1, inplace = True) 
d.drop("neckline_shape",axis = 1, inplace = True) 
d.drop("wearing_jacket",axis = 1, inplace = True) 
d.drop("wearing_hat",axis = 1, inplace = True)
d.drop("wearing_glasses",axis = 1, inplace = True) 

# get rid of empty rows
colorData = d[d.major_color != 'empty']
colorData = colorData[colorData.classes != 'empty']
colorData = colorData[colorData.clothing_pattern != 'empty']

# Plot distribution of colors
fig_dims = (2,2)
plt.subplot2grid(fig_dims, (1, 1))
colorData['major_color'].value_counts().plot(kind='bar', 
                                    title='major_colors')# 11 classes
#visualize types of clothing
fig_dims = (2,2)
plt.subplot2grid(fig_dims, (1, 1))
d['classes'].value_counts().plot(kind='bar', 
                                    title='Types of clothing')# 7 classes
#Plot the distribution of the pattern data
patternData = d[d.clothing_pattern != 'empty']
patternData = patternData[patternData.major_color != 'empty']
patternData = patternData[patternData.classes != 'empty']
plt.subplot2grid(fig_dims, (1, 1))
patternData['clothing_pattern'].value_counts().plot(kind='bar', 
                                    title='clothing_pattern')# 6 classes

"*********** TF Records for tensorflow Object detection API *******"
"""
    This part of the module for creating custom tfRecords
"""


#keep dropping since you obtained your data
d.drop("clothing_pattern",axis = 1, inplace = True) 
d.drop("major_color",axis = 1, inplace = True) 

#do the split and save your csv files into the data directory after for generating tfRecords
train_labels, test_labels = train_test_split(d, test_size=0.1)
train_labels.to_csv('data/train_labels.csv')
test_labels.to_csv('data/test_labels.csv')

"******************************************************************"
"HELPER FUNCTIONS"
 
"Function which gives the complete path given location in csv (according to structure of streetsyle27k dataset)"
#exemple filename: str(d['filename'][7])
def return_path(filename):
    first = filename[0]
    second = filename[1]
    third = filename[2]
    path = "images/streetstyle27k" + "/" + first + "/" + second + "/" + third + "/" + str(filename)
    return path

"Function matching filename with an image and showing that image"
import os
def find_image(filename):
    first = filename[0]
    second = filename[1]
    third = filename[2]
    os.chdir("images/streetstyle27k" + "/" + first + "/" + second + "/" + third)
    myImage = PIL.Image.open(filename)
    myImage.load()
    imshow(myImage)
    #get back to the current working directory
    os.chdir("../../../../..")
    return myImage
#exemple input
#find_image(str(d['filename'][7]))

"Function for transferring clothing_patterns into another directory "
#margin for taking only the clothes part of the dataset images and not the heads
marginHead = 60
margin = 10

#Divide the data into 2 parts %90 train data and %10 test data
train_labels_pattern, test_labels_pattern = train_test_split(patternData, test_size=0.1)
def dump_training_pattern_data():
    for index, item in train_labels_pattern.iterrows():
        print(item['filename'])
        image = find_image(item['filename'])
        print(os.getcwd())
        #crop the image from the bounding boxes
        t = image.crop((item['xmin'] + margin, item['ymin'] + marginHead, item['xmax'] - margin, item['ymax'] - margin)) 
        os.chdir('data/transfer_learning/pattern/training_set ') 
        os.chdir(str(item['clothing_pattern']))
        t.save(fp = item['filename'])
        os.chdir("../../../../..")
def dump_test_pattern_data():
    for index, item in test_labels_pattern.iterrows():
        print(item['filename'])
        image = find_image(item['filename'])
        print(os.getcwd())
        marginHead = 20
        margin = 10
        #crop the image from the bounding boxes
        t = image.crop((item['xmin'] + margin, item['ymin'] + marginHead, item['xmax'] - margin, item['ymax'] - margin)) 
        os.chdir('data/transfer_learning/pattern/test_set ') 
        os.chdir(str(item['clothing_pattern']))
        t.save(fp = item['filename'])
        os.chdir("../../../../..")

#prepare train and test data for pattern    
dump_training_pattern_data()  
dump_test_pattern_data()


"Function for transferring major_colors into another directory for trainig and test data" 
train_labels_colors, test_labels_colors = train_test_split(colorData, test_size=0.1)
def dump_training_color_data():
    for index, item in train_labels_colors.iterrows():
        print(item['filename'])
        image = find_image(item['filename'])
        print(os.getcwd())
        #crop the image from the bounding boxes
        t = image.crop((item['xmin'], item['ymin']+50, item['xmax'], item['ymax'])) 
        os.chdir('data/transfer_learning/colors/training_set ') 
        os.chdir(str(item['major_color']))
        t.save(fp = item['filename'])
        os.chdir("../../../../..")
def dump_test_color_data():
    for index, item in test_labels_colors.iterrows():
        print(item['filename'])
        image = find_image(item['filename'])
        print(os.getcwd())
        #crop the image from the bounding boxes
        t = image.crop((item['xmin'] + margin, item['ymin'] + marginHead, item['xmax'] - margin, item['ymax'] - margin)) 
        os.chdir('data/transfer_learning/colors/test_set ') 
        os.chdir(str(item['major_color']))
        t.save(fp = item['filename'])
        os.chdir("../../../../..")
        
#prepare train and test data for color       
dump_training_color_data()  
dump_test_color_data()

"Function for transferring classes into another directory for trainig and testing " 
#first cleanise the data from empty values in colors and pattern and clothes
d = d[d.clothing_pattern != 'empty']
d = d[d.major_color != 'empty']
d = d[d.classes != 'empty']
train_labels_clothes, test_labels_clothes = train_test_split(d, test_size=0.1)
      
def dump_clothing_training_data():
    for index, item in train_labels_clothes.iterrows():
        print(item['filename'])
        image = find_image(item['filename'])
        print(os.getcwd())
        #crop the image from the bounding boxes
        t = image.crop((item['xmin'] + margin, item['ymin'] + marginHead , item['xmax'] - margin, item['ymax'] - margin)) 
        os.chdir('data/transfer_learning/clothes/training_set ') 
        os.chdir(str(item['classes']))
        t.save(fp = item['filename'])
        os.chdir("../../../../..")
        
def dump_clothing_test_data():
    for index, item in test_labels_clothes.iterrows():
        print(item['filename'])
        image = find_image(item['filename'])
        print(os.getcwd())
        #crop the image from the bounding boxes
        t = image.crop((item['xmin'], item['ymin'], item['xmax'], item['ymax'])) 
        os.chdir('data/transfer_learning/clothes/test_set ') #Incomplete make sure making test/train split
        os.chdir(str(item['classes']))
        t.save(fp = item['filename'])
        os.chdir("../../../../..")
        
#prepare train and test data for clothing classes
dump_clothing_training_data() 
dump_clothing_test_data() 

"*****************BOUNDING BOX DRAWING FUNCTION*******************************"
import cv2 as cv


#function for drawing a rectangle on the image and writing the prediction with probability score
#parameters: path, resultlabel, coordinates of the found position with two tupels, number of the image
def custom_bbox(path, predictions, xmin, ymin, xmax, ymax):
    img = cv.imread(str(path))
    img = cv.resize(img, (640, 640))
    img = cv.rectangle(img,(xmin,ymin),(xmax,ymax), (0,255,0), 4)
    font = cv. FONT_HERSHEY_COMPLEX_SMALL
    cv.putText(img,'T-shirt',(int(xmin),int(ymax)), font, 1,(255,255,255),2,cv.LINE_AA)
    cv.imwrite("data/annoted_images/img_final.jpg", img)
    


"************************* TRANSFER LEARNING *************"  
"helper functions"
"""
#to create directories datasets with non-empty patterns and colors
transferData = dataset[dataset.clothing_pattern and dataset.colors != 'empty']  

"Transfer learning on a pretrained Convolutional Neural Network with Keras for pattern recognition"
#import necessary libraries
import keras
import tensorflow as tf
import seaborn as snsfrom 
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

#load the modal
model = VGG16(include_top=True, weights='imagenet')
input_shape  = model.layers[0].output_shape[1:3]

#overview 
model.summary()
#cut the layer prevent the output
transfer_layer = model.get_layer('block5_pool') #try cutting different places 
#weights 
transfer_layer.output
#new model
conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)
# Keras Sequential model.
new_model = Sequential()

# Add the convolutional part of the VGG16 model .
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense layer
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# dropout
new_model.add(Dropout(0.5))

# classification layer.
num_classes = 7
new_model.add(Dense(num_classes, activation='softmax'))

#adam optimizer
optimizer = Adam(lr=1e-5)

loss  = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))
print_layer_trainable()
#now disable all other layers beacuse you wont want to retrain previous layers,
conv_model.trainable = False
for layer in conv_model.layers[:9]:
    layer.trainable = False

print_layer_trainable()
#compile the model
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#epochs and steps
epochs = 20
steps_per_epoch = 100

#do the data preprocessing direct images to the directories where they belong

train_path = 'data/transfer_learning/clothes/training_set'
test_path = 'data/transfer_learning/clothes/test_set'

train_batches = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True).flow_from_directory(train_path, target_size = (224,224), classes = ['Dress', 'Outerwear', 'Shirt', 'Suit', 'Sweater', 'Tank top', 'T-shirt'], batch_size = 20)
test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test_path, target_size = (224,224), classes = ['Dress', 'Outerwear', 'Shirt', 'Suit', 'Sweater', 'Tank top', 'T-shirt'], batch_size = 20)

#fit the model
new_model.fit_generator(
        train_batches,
        steps_per_epoch=250, #4000
        epochs=2, #200
        validation_data=test_batches,
        validation_steps=100) #400
new_model.save_weights('clothes.h5') #saving weights

#make a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('{{filename}}', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = new_model.predict(test_image)
train_batches.class_indices
if result[0][0] == 1:
    prediction = 'Dress'
    print(prediction)
elif(result[0][1] == 1):
    prediction = 'Outerwear'
    print(prediction)
elif(result[0][2] == 1):
    prediction = 'Shirt'
    print(prediction)
elif(result[0][3] == 1):
    prediction = 'Suit'
    print(prediction)
elif(result[0][4] == 1):
    prediction = 'Sweater'
    print(prediction)
elif(result[0][5] == 1):
    prediction = 'Tank top'
    print(prediction)
elif(result[0][6] == 1):
    prediction = 'T-shirt'
    print(prediction)

"""
"****************************FOR CUSTOM TF RECORDS DO NOT USE OTHERWİSE**************************"
"Experimental Neural Network tfRecord generation"
d1 = d[d.filename.str.contains('^0')] #starts with zeros so can be used
#saving
train_labels, test_labels = train_test_split(d1, test_size=0.1)
train_labels.to_csv('data/train_labels.csv')
test_labels.to_csv('data/test_labels.csv')

#run  python train.py --logtostderr --train_dir = training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config



