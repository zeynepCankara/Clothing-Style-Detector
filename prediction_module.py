# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 08:56:06 2018

@author: Zeynep CANKARA

Detection module
"""
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
#make a prediction from your test
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv
import os
import json

"""
    Read the json data for test

json_data=open('data.txt').read()
data = json.loads(json_data)
print(data)
"""

"""
    Function for cropping the image and detecting the clothing type, pattern, color on the image if exist
    the images will later saved with the bounding box and prediction 
    param: path, type = str, filepath of the image,
        xmin, ymin, xmax, ymax, type = int, bounding-box coordinates,
        im_width, im_height, type = int, image dimensions width x height
"""

def crop_image(path,xmin, ymin, xmax, ymax, im_width, im_height):
    myImage = cv.imread(str(path))
    myImage = cv.resize(myImage,(im_width,im_height))
    cropped = myImage[ymin: ymax, xmin:xmax]
    cv.imwrite("prediction/img_trial2.jpg", cropped)
    croped_predictions = clothing_color_pattern("prediction/img_trial2.jpg")
    #display the predictions on the console
    print(croped_predictions)
    if(str(croped_predictions) != ""):
        custom_bbox(str(path),str(croped_predictions) ,xmin ,  ymin, xmax, ymax, im_width, im_height)
    return croped_predictions

"""
    Function for drawing a bounding box on the image
    param: path, type = str, filepath of the image,
        predictions, type = str, 
        xmin, ymin, xmax, ymax, type = int, bounding-box coordinates,
        im_width, im_height, type = int, image dimensions width x height        
"""
def custom_bbox(path, predictions, xmin, ymin, xmax, ymax, im_width, im_height):
    img = cv.imread(str(path))
    img = cv.resize(img, (im_width, im_height))
    print(im_width)
    print(im_height)
    #drawing a rectangle on the image on the place where clothes detected
    img = cv.rectangle(img,(xmin,ymin),(xmax,ymax), (0,255,0), 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    #font size set according to the image size
    font_size = [1, 0.75, 0.5, 0.25, 0.10, 0.05]
    if((im_width * im_height) > 1638400 ):
        font_size = float(font_size[0])
    elif((im_width * im_height) > 409600):
        font_size = float(font_size[1])    
    elif((im_width * im_height) > 102400):
        font_size = float(font_size[2])
    elif((im_width * im_height) > 25600):
        font_size = float(font_size[3])
    elif((im_width * im_height) > 6400):
        font_size = float(font_size[4])  
    else:
        font_size = float(font_size[5])
    #writing the detection result on the bounding-box
    cv.putText(img,str(predictions),(int(xmin),int(ymax)), font, float(font_size) ,(0,0,0),2,cv.LINE_AA)
    cv.imwrite(str(path), img)

"""
    Function which loads models and performs the detection on the cropped section of the image
    param: path, type = str
"""

def clothing_color_pattern(path):
    #the dictionary for evaluation
    valid_classes = {'T-shirt': ['jersey', 'T-shirt', 'tee shirt'], 'Dress':['dress', 'gown', 'overskirt', 'hoopskirt', 'stole', 'abaya', 'academic_gown', 'poncho', 'breastplate'], 'Outerwear':['jacket', 'raincoat', 'trench coat','book jacket', 'dust cover', 'dust jacket', 'dust wrapper', 'pitcher'], 'Suit':['suit','bow tie', 'bow-tie', 'bowtie','suit of clothes'], 'Shirt':['shirt'], 'Sweater':['sweater', 'sweatshirt','bulletproof_vest', 'velvet'] , 'Tank top':['blause', 'tank top', 'maillot', 'bikini', 'two-piece', 'swimming trunks', 'bathing trunks'], 'Skirt':['miniskirt', 'mini']}
    have_glasses = {'Glasses': ['glasses', 'sunglass', 'sunglasses', 'dark glasses','shades']}
    wear_necklace = {'Necklace': ['neck_brace','necklace']}
    
    #initializing the prediictions
    prediction_color_clothes = ""
    acsessories = ""
    clothing_type = ""
    
    #LOADING MODALS
    #Load the ResNet50 model
    resnet_model = resnet50.ResNet50(weights='imagenet')
    #load pattern model
    pattern_model = load_model('pattern.h5')
    #load color model
    color_model = load_model('color.h5')
    
    #run model for color resNet class detection:
     #process for the resNet model
    test_image_resnet = image.load_img(path, target_size = (224, 224))
    test_image_resnet = image.img_to_array(test_image_resnet)
    
    #plot the image for test
    plt.imshow(test_image_resnet/255.)
    
    test_image_resnet = np.expand_dims(test_image_resnet, axis = 0)
    result_resnet = resnet_model.predict(test_image_resnet)
    label = decode_predictions(result_resnet)

    #predictions by resnet
    print(label[0])
    print(label[0][0])
    #check is prediction matches
    for element in range(len(label[0])):
        for key in valid_classes:
            if(label[0][element][1] in valid_classes[key]):
                if(float(label[0][element][2]) >= 0.055):
                    if(clothing_type == ""):
                        clothing_type = str(key)
                        break
                    
    #check for acsessories
    for element in range(len(label[0])):
        for key in have_glasses:
            if(label[0][element][1] in have_glasses[key]):
                if(float(label[0][element][2]) >= 0.04):
                    acsessories += str(key) + ","
    
    for element in range(len(label[0])):
        for key in wear_necklace:
            if(label[0][element][1] in wear_necklace[key]):
                if(float(label[0][element][2]) >= 0.05):
                    acsessories += str(key) + " "   
    
    #prepare the input image
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #predicting the pattern and color
    result_pattern = pattern_model.predict_classes(test_image)
    result_color = color_model.predict_classes(test_image)

                    
    #check the pattern   
    pattern_classes = ['Floral','Graphics','Plaid','Solid','Spotted','Striped']            
    prediction_pattern = pattern_classes[int(result_pattern)]
    
    #check the color
    color_classes =  ['Black', 'Blue', 'Brown', 'Cyan', 'Gray', 'Green', 'More than 1 color', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    prediction_color = color_classes[int(result_color)]


    #add the pattern info to the prediction
    prediction_color_clothes += str(prediction_pattern) + " , " + str(prediction_color)
    
    if((acsessories == "") and (clothing_type == "")):
        return str(prediction_color_clothes)
    else:
        return str(acsessories) + " " + str(prediction_color_clothes) + " " + str(clothing_type)



"""
    Takes the prediction of the acsessories if the class prediction in the acsessories 
    param: acsessories_class prediction of the acsessoies in the acsessories dictionary
    outputs the prediction, later to be used in custom_bbox()
    
    things to note: acsessories classes do not match with imagenet classes but match with 
    pbtxt file which you direct your model
    you can change this if you change type of your model from (model_zoo)
    COCO is good at people detection
"""

def acsessory_pattern_color(acsessories_class, path, xmin, ymin, xmax, ymax, im_width, im_height):
    myImage = cv.imread(str(path))
    myImage = cv.resize(myImage,(im_width,im_height))
    cropped = myImage[ymin: ymax, xmin:xmax]
    cv.imwrite("prediction/img_trial2.jpg", cropped)
    #load pattern model
    pattern_model = load_model('pattern2.h5')
    #load color model
    color_model = load_model('color.h5')
    #Process for custom_color_classifier
    test_image = image.load_img(str("prediction/img_trial2.jpg"), target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #predicting the pattern
    result_pattern = pattern_model.predict_classes(test_image)
    #result_color = color_model.predict_classes(test_image)
    
    #process for custom_pattern_classifier
    test_image2 = image.load_img(str("prediction/img_trial2.jpg"), target_size = (128, 128))
    test_image2 = image.img_to_array(test_image2)
    test_image2 = np.expand_dims(test_image2, axis = 0)    
    result_color = color_model.predict_classes(test_image2)
    #check the color
    color_classes =  ['Black', 'Blue', 'Brown', 'Cyan', 'Gray', 'Green', 'More than 1 color', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    prediction_color = color_classes[int(result_color)]
    
    #check the pattern    
    pattern_classes = ['Floral', 'Graphics', 'Plaid', 'Solid', 'Spotted', 'Striped']    
    prediction_pattern = pattern_classes[int(result_pattern)]       

    prediction_for_color_pattern =  str(prediction_color) + ", "  + str(prediction_pattern) + " " +  str(acsessories_class) 
    custom_bbox(str(path),str(prediction_for_color_pattern) ,xmin ,  ymin, xmax, ymax, im_width, im_height)
    
"""
    Main function for reading the json data
    param: data json 
"""
def read_json_data(data):
    acsessories_list = ["b'backpack", "b'umbrella", "b'book", "b'cell phone", "b'tie", "b'suitcase", "b'handbag", "b'baseball glove", "b'tennis racket", "b'laptop" ]
    for element in data:
        current_image = element
        im_dictionary = data[str(current_image)]
        im_width = im_dictionary['width']
        im_height = im_dictionary['height']
        file_path = im_dictionary['file_path']
        box = im_dictionary['boxes']
        for index in range(len(box['classes'])):
            if(box['classes'][index] == "b'person'"):
                #take the bounding box on the image
                xmin = box['xmin'][index]
                ymin = box['ymin'][index]
                xmax = box['xmax'][index]
                ymax = box['ymax'][index]
                scores = box['scores'][index]
                #train with your own classifiers
                crop_image(str(file_path),xmin, ymin, xmax, ymax, im_width, im_height)
            elif(box['classes'][index] in acsessories_list):
                #take the bounding box on the image
                print(box['classes'][index])
                xmin = box['xmin'][index]
                ymin = box['ymin'][index]
                xmax = box['xmax'][index]
                ymax = box['ymax'][index]
                scores = box['scores'][index]
                #train with your own classifiers
                acsessory_pattern_color(str(box['classes'][index][2:]), str(file_path),xmin, ymin, xmax, ymax, im_width, im_height)

#test
#read_json_data(data)
#print(data)