# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:57:19 2018

@author: Zeynep Cankara

Playing with pre-trained models and constructing keras models

fine-tune model: good article: https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
 
"""
#ALL LIBRARIES IMPORTED IN HERE
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv
from flask import Flask, current_app

" PRE TRAINED MODELS "
#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')
 
#Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
 
#Load the ResNet50 model
resnet_model = resnet50.ResNet50(weights='imagenet')
 
#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

#make a prediction from your test
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
import matplotlib.pyplot as plt

test_image = image.load_img('{{filename}}', target_size = (224, 224))
plt.imshow(test_image)
plt.show()
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = inception_model.predict(test_image)
result2 = vgg_model.predict(test_image)
result3 = resnet_model.predict(test_image) #most accurate in my case RES_NET MODEL
result4 = mobilenet_model.predict(test_image)

label = decode_predictions(result)
label2 = decode_predictions(result2)
label3 = decode_predictions(result3)
label4 = decode_predictions(result4)
print(label)
print(label2)
print(label3)
print(label4)



print(label3[0][0][2]) #gives highest probability
print(label3[0][0][1]) #gives the class

"*********************  Classifier for recognizing clothing category ****************************"
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
num_classes = 7
classifier.add(Dense(units = num_classes, activation = 'softmax'))


# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
train_path = 'data/transfer_learning/clothes/training_set'
test_path = 'data/transfer_learning/clothes/test_set'
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (64, 64),
                                                 classes = ['Dress', 'Outerwear', 'Shirt', 'Suit', 'Sweater', 'Tank top', 'T-shirt'],
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (64, 64),
                                            classes = ['Dress', 'Outerwear', 'Shirt', 'Suit', 'Sweater', 'Tank top', 'T-shirt'],
                                            batch_size = 32)
checkpointer = ModelCheckpoint(filepath='classifier2.h5', verbose=1, save_best_only=True)

#if you have GPU supported machine you are encouraged to train with more epochs and steps
classifier.fit_generator(training_set,
                         steps_per_epoch = 9000, 
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 2000,
                         callbacks=[checkpointer])
classifier.save_weights('classifier_clothes.h5')

"************************** single prediction *****************************"
test_image = image.load_img('{{filename}}', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result2 = classifier.predict(test_image)
result = classifier.predict_classes(test_image)
predictions = classifier.predict(test_image)
classes = np.argmax(predictions, axis=1)


"******************************************** BOUNDING BOX *****************************************"
def custom_bbox(path, predictions, xmin, ymin, xmax, ymax):
    img = cv.imread(str(path))
    img = cv.resize(img, (640, 640))
    img = cv.rectangle(img,(xmin,ymin),(xmax,ymax), (0,255,0), 4)
    font = cv. FONT_HERSHEY_COMPLEX_SMALL
    cv.putText(img,predictions,(int(xmin),int(ymax)), font, 1,(0,0,0),2,cv.LINE_AA)
    cv.imwrite("data/annoted_images/img_final2.jpg", img)

"********************** MAIN FUNCTION FOR CLOTHES DETECTION ***********************************"
"double models for detecting the clothing category of the image"
#takes the path of the image and returns the text classified


def clothing_category(path):
    custom_model = load_model('classifier2.h5')
    resnet_model = resnet50.ResNet50(weights='imagenet')
    pattern_model = load_model('pattern.h5')
    
    #the dictionary for evaluation
    valid_classes = {'T-shirt': ['jersey'], 'Dress':['dress', 'gown', 'overskirt', 'hoopskirt', 'stole', 'abaya', 'academic_gown', 'poncho'], 'Outerwear':['jacket', 'raincoat', 'trench coat','book jacket', 'dust cover', 'dust jacket', 'dust wrapper', 'pitcher'], 'Suit':['suit','bow tie', 'bow-tie', 'bowtie','suit of clothes'], 'Shirt':['shirt'], 'Sweater':['sweater', 'sweatshirt','bulletproof_vest', 'velvet'] , 'Tank top':['blause', 'tank top', 'maillot', 'bikini', 'two-piece', 'swimming trunks', 'bathing trunks']}
    have_glasses = {'Glasses': ['glasses', 'sunglass', 'sunglasses', 'dark glasses','shades']}
    wear_necklace = {'Necklace': ['neck_brace']}
    #Process for custom_classifier
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #result = custom_model.predict_classes(test_image)
    #predicting the pattern
    result_pattern = pattern_model.predict_classes(test_image)

    #process for the resNet model
    test_image_resnet = image.load_img(path, target_size = (224, 224))
    test_image_resnet = image.img_to_array(test_image_resnet)
    test_image_resnet = np.expand_dims(test_image_resnet, axis = 0)
    result_resnet = resnet_model.predict(test_image_resnet)
    label = decode_predictions(result_resnet)
    print(label[0])
    print(label[0][0])
    prediction = ""
    acsessories = ""
    for element in range(len(label[0])):
        for key in have_glasses:
            if(label[0][element][1] in have_glasses[key]):
                if(float(label[0][element][2]) >= 0.05):
                    acsessories += str(key) + ","
    
    for element in range(len(label[0])):
        for key in wear_necklace:
            if(label[0][element][1] in wear_necklace[key]):
                if(float(label[0][element][2]) >= 0.05):
                    acsessories += str(key) + ","
    if(result_pattern == 0):
        prediction_pattern = 'Floral'
    elif(result_pattern == 1):
        prediction_pattern = 'Graphics'
    elif(result_pattern == 2):
        prediction_pattern = 'Plaid'
    elif(result_pattern == 3):
        prediction_pattern = 'Solid'
    elif(result_pattern == 4):
        prediction_pattern = 'Spotted'
    elif(result_pattern == 5):
        prediction_pattern = 'Striped'
    else:
        pass
    
    for element in range(len(label[0])):
            for key in valid_classes:
                if(label[0][element][1] in valid_classes[key]):
                    if(float(label[0][element][2]) >= 0.09):
                        prediction += acsessories + str(key) + " , " + str(prediction_pattern) 
                        break
                    
 
    return prediction

   
"**********************************Color Recognizer Test****************************"
color_model = load_model('color.h5')
path = 'data/transfer_learning/colors/training_set/Blue/0aac88cb48b95fa4a5a238191e74b9ab_506979550133139012_344816196.jpg'
test_image = image.load_img(path, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
y_prob = color_model.predict_classes(test_image)
y_classes = y_prob.argmax(axis=-1)
"***********************************************************************************"
"Small neural network for color deyection"
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
num_classes = 13
classifier.add(Dense(units = num_classes, activation = 'softmax'))


# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
train_path = 'data/transfer_learning/colors/training_set'
test_path = 'data/transfer_learning/colors/test_set'
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (64, 64),
                                                 classes = ['Black', 'Blue', 'Brown', 'Cyan', 'Gray', 'Green', 'More than 1 color', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'],
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (64, 64),
                                            classes = ['Black', 'Blue', 'Brown', 'Cyan', 'Gray', 'Green', 'More than 1 color', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'],
                                            batch_size = 32)
checkpointer = ModelCheckpoint(filepath='color.h5', verbose=1, save_best_only=True)
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 14,
                         validation_data = test_set,
                         validation_steps = 2000,
                         callbacks=[checkpointer])
classifier.save_weights('color_last.h5')
"***********************************************************************************"
"Small neural network for pattern recognition"
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
num_classes = 6
classifier.add(Dense(units = num_classes, activation = 'softmax'))


# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
train_path = 'data/transfer_learning/pattern/training_set'
test_path = 'data/transfer_learning/pattern/test_set'
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (128, 128),
                                                 classes =  ['Floral', 'Graphics', 'Plaid', 'Solid', 'Spotted', 'Striped'],
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (128, 128),
                                            classes = ['Floral', 'Graphics', 'Plaid', 'Solid', 'Spotted', 'Striped'],
                                            batch_size = 32)
checkpointer = ModelCheckpoint(filepath='pattern.h5', verbose=1, save_best_only=True)
classifier.fit_generator(training_set,
                         steps_per_epoch = 5000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000,
                         callbacks=[checkpointer])
classifier.save_weights('pattern_last.h5')


#NOTE: you're encouraged to use classifier weights of the checkpoints for more accurate results...


