
# Import required packages
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv

# Handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = ap.parse_args()
                     
# Loading Pre_trained model
model = load_model('my_model_weights.h5')

# Reading the input image
image = cv2.imread(args.image)
if image is None:
    print("Could not read input image")
    exit()

# Detecting faces in the image
face, confidence = cv.detect_face(image)
classes = ['man','transgender','woman']

# Looping through detected faces  in the image
for idx, f in enumerate(face):

     # get corner points of a face as rectangle       
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]
    # draw rectangle over face
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

    # crop the detected face region
    face_crop = np.copy(image[startY:endY,startX:endX])

    # Preprocessing the face for Recognising the gender 
    face_crop = cv2.resize(face_crop, (256,256))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # Predicting the preprocessed face for gender detection 
    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    # get label with max probability
    idx = np.argmax(conf)
    label = classes[idx]

    label = "{}: {:.2f}%".format(label, conf[idx] * 100)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # write label and confidence above face rectangle
    cv2.putText(image, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

# display output
cv2.imshow("gender detection", image)

# press any key to close window           
cv2.waitKey()

# save output
cv2.imwrite("gender_detection.jpg", image)

# release resources
cv2.destroyAllWindows()