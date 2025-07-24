# Suppress TensorFlow warnings and info messages
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Available backend options are: "jax", "torch", "tensorflow".
os.environ["KERAS_BACKEND"] = "tensorflow"

os.system("clear")

import cv2
import time

import os
import json
import keras
import numpy as np
import matplotlib.pyplot as plt

from functions import *

# load the facenet128 model
@keras.saving.register_keras_serializable()
def scaling(x, scale=1.0):
	return x * scale
model = keras.saving.load_model("hf://logasja/FaceNet")

# load model to boost performance
test = model.predict(np.zeros((1, 160, 160, 3)))

# Access camera
# cap = getCamera(width= 640, height=480)
cap = getCamera()

DETECTION_MODE = "all"

label = ""
distance = 0

while cap.isOpened():
    # assign the key and read every 1ms
    key = cv2.waitKey(1) & 0xFF
    
    # read the frame
    # ret   : boolean
    # frame : numpy array
    _, frame = cap.read()
    
    # face detection
    faceMaps = faceDetection(frame, mode=DETECTION_MODE)
    # debug
    # print(type(faceMaps), len(faceMaps),"\n", faceMaps)
    
    # process detected faces
    for face_region in faceMaps:
        # drawRectangle(face, frame)
        
        # extract the face region from the frame
        x, y, w, h = face_region[0], face_region[1], face_region[2], face_region[3]
        face = frame[y+1:y+h, x+1:x+w]
        
        # save face image when pressing enter
        # if key == ord('\r'):
        #     saveFaceImage(label="yasir", face=face)
        
        # face recognition
        label, distance = faceRecognition(face, threshold=10.0)
        drawRectangle(face=face_region, frame=frame, label=label, distance=distance)
        # debug
        # print(label, distance)
    
    # display log/information
    cv2.putText(frame, f"Detected Mode : {DETECTION_MODE}", (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Detected Faces : {len(faceMaps)}", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"First Face : {label, distance}", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Resolution : {frame.shape}",  (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # display camera streaming
    cv2.imshow("Face Recognition", frame)
    
    # reduce cpu usage
    time.sleep(0.01)
    
    # press q for exit the loop
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()