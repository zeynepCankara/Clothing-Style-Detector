# clothing_detector
Clothes detector, predicting patterns and color together with the clothing category. Made for my internship Summer 2018 @Somera. 

*Exemple Detections*

<img src="test_images/img_output4.jpg" width="400" height="400">

<img src="/test_images/image7.jpg" width="400" height="400">

<img src="/test_images/image_dress.jpg" width="400" height="400">

<img src="/test_images/image_dress2.jpg" width="400" height="550">

Code uses Keras API Tensorflow as backend.

* In order to run the code Tensorflow Object Detection API set-up needed.
* Copy and paste all the files in to the tensorflow/models/reserach/object_detection directory

* Dataset taken from: http://streetstyle.cs.cornell.edu/
 * clothes_detector.py (uses tenorflow object-detection API and reads images from static/img folder outputs the prediction in train_images  folder)
 * train_test_splitter.py (clean, visualize the dataset dataset ready to feed into Keras classifiers)
 * prediction_module.py (runs custom classifiers on Tensorflow Object detection API's region proposals)
 * keras_models.py (creates color, pattern, clothing classifiers uses Keras API)
