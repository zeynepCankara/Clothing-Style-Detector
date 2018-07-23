# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:57:19 2018

@author: Zeynep Cankara

Playing with pre-trained models

Note you should try to fine-tune model with new set of classes: good article: https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
 
"""
#ALL LIBRARIES IMPORTED IN HERE
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
from flask import Flask, current_app
import json

"""
    functions in the final_prediction_zeynep module
"""
from prediction_module import crop_image, custom_bbox, clothing_color_pattern, acsessory_pattern_color


import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
"*********************************************************************************"
#map imports
from utils import label_map_util
from utils import visualization_utils as vis_util
"*********************************************************************************"
# What model to download.
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

#number of classes in the COCO dataset
NUM_CLASSES = 90
"*********************************************************************************"
#set the model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
"********************************************************************************"
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
"*********************************************************************************"
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
"**********************************************************************************"
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
"************************************************************************************"
PATH_TO_TEST_IMAGES_DIR = 'static/img'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, int(len(PATH_TO_TEST_IMAGES_DIR))) ] #runs prediction on the every image in the directory
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
"**********************************************************************************"
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        #small test
        print( detection_masks_reframed )
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

"**************************************************************"
def predict_image():
    #initialize the dictionary
    image_json_dict = dict()
    #just to label images
    count = 0
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      """
      Small trial for saving the image
      """
      #save masked images 'prediction/image_path
      plt.imsave("prediction/" + str(str(image_path)[33:-4] + "masked.jpg" ),image_np)
      #cv.imwrite(str(str(image_path) + "masked"), image_np)
      "************************************************************"
      boxes = output_dict['detection_boxes']
      #initialize the lists here for bounding boxes not real one playing with format
      image_dict = dict()
      xmin_list = list()
      xmax_list = list()
      ymin_list = list()
      ymax_list = list()
      class_list = list()
      score_list = list()
      box_dict = dict()
      prediction_dict = dict()
      acsessory_list = list()
      clothing_list = list()
      #iterating over possible boxes
      for i in range(min(20, boxes.shape[0])):
        if output_dict['detection_scores'] is None or output_dict['detection_scores'][i] > 0.5:
          box = tuple(boxes[i].tolist())
          
          ymin, xmin, ymax, xmax = box
          im_width, im_height = image.size
          (left, right, top, bottom) = (xmin * im_width, xmax * im_width, 
                                      ymin * im_height, ymax *im_height)
          print("for image: " + str(i) + ": " + str(left) +" , " + str(right) +" , " + str(output_dict['detection_classes'][i]) )
          objects = []
          for index, value in enumerate(output_dict['detection_classes']):
              object_dict = {}
              threshold = 0.5
              if output_dict['detection_scores'][index] > threshold:
                  object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                            output_dict['detection_scores'][index]
                  objects.append(object_dict)
          #obtaining classes
          print(str(list(objects[i].keys())[0])) 
          #obtaining class scores
          print(float(list(objects[i].items())[0][1]))
          #gives the label of the detection you can get the score by .values()
          #add your boxes for constructing a json 
          class_list.append(str(list(objects[i].keys())[0]))
          xmin_list.append(int(left))
          ymin_list.append(int(top))
          xmax_list.append(int(right))
          ymax_list.append(int(bottom))
          score_list.append(float(list(objects[i].items())[0][1]))
          #I decided doing the saving here
          clothing = ""
          acsessory = ""
            #load image and pattern models here 
          im_width, im_height = image.size
          if(str(list(objects[i].keys())[0]) == "b'person'"):
              clothing = crop_image(str(image_path),int(left), int(top), int(right), int(bottom), im_width, im_height)
          acsessories_list = ["b'backpack", "b'umbrella", "b'book", "b'cell phone", "b'tie", "b'suitcase", "b'handbag", "b'baseball glove", "b'tennis racket", "b'laptop" ]
          if(str(list(objects[i].keys())[0]) in acsessories_list):
              acsessory = acsessory_pattern_color(str(list(objects[i].keys())[0]), str(image_path),int(left), int(top),  int(right) ,int(bottom), im_width, im_height)
           #End of my experiment
          clothing_list.append(str(clothing))
          acsessory_list.append(str(acsessory))
      prediction_dict = {'acsessories': acsessory_list, 'clothes': clothing_list}
      #WORKING PLACE  
      box_dict = {'classes': class_list, 'xmin': xmin_list, 'ymin':  ymin_list, 'xmax': xmax_list, 'ymax':  ymax_list, 'scores' : score_list }  
      image_dict['boxes'] = box_dict
      image_dict['prediction result'] = prediction_dict
      im_width, im_height = image.size
      (left, right, top, bottom) = (xmin * im_width, xmax * im_width, 
                                  ymin * im_height, ymax *im_height)
      image_dict['file_path'] = str(image_path)
      image_dict['width'] = int(im_width)
      image_dict['height'] = int(im_height)
      ymin, xmin, ymax, xmax = box
      #image_list.append(image_dict)  
      image_json_dict['image' + str(count)] = image_dict
      count+=1
    return image_json_dict

json_dict = predict_image()  

"""
    Writing the json as a txt file named 'data.txt'
"""
with open('data.txt', 'w') as outfile:
    json.dump(json_dict, outfile)



"""
    Overlay 2 images on top of each other, for applying mask on the custom prediction 
"""
#test can applied to all images
"""
image2 = cv.imread("test_images/image4.jpg")
image = cv.imread("test_images/image4masked.jpg")

overlay = image.copy()
output = image2.copy()
cv.addWeighted(overlay, 0.5, output, 0.5,
	0, output) #need to be same shape
	# show the output image
cv.imwrite("test_images/img_output4.jpg", output)
"""


