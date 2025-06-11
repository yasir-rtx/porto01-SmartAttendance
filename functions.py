import os
import cv2
import json
import time
import keras
import numpy as np

# Load a model from Hugging Face Hub
# The model is a FaceNet512 model trained on the VGGFace2 dataset.
# It can be used for face recognition tasks.
# The model is saved in the TensorFlow format.
# The custom object 'scaling' is used to scale the input images.
# The model can be used to extract face embeddings from images.
@keras.saving.register_keras_serializable()
def scaling(x, scale=1.0):
	return x * scale
# Load the FaceNet512 model from Hugging Face Hub
model = keras.saving.load_model(
	"hf://logasja/FaceNet512",
	custom_objects={"scaling": scaling}
)

# Pre Predict function to boostup the performance
def preload():
    # Preload the model to boostup the performance
    imgTest = np.zeros((1, 160, 160, 3))
    signature = model.predict(imgTest)
    print("Model preloaded successfully.")
    return signature

# Access camera
def setCamera(index, width=640, height=480, fps=30):
    cap = cv2.VideoCapture(index)
    # Set the camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Set the camera frame rate (optional)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

# Change the camera index if necessary
def getCamera(index=1, max_attempt=2):
    # Try to open external cameras
    try:
        while max_attempt:
            print(f"Opening Camera {index}...")
            cap = setCamera(index)
            if not cap.isOpened():
                print(f"Camera {index} is not Available!")
                index += 1
                max_attempt -= 1
            else:
                print(f"Camera {index} is Available!")
                return cap
    except Exception as e:
        print(f"An error occurred while accessing the camera: {e}")
        exit()
    # Open default camera if external cameras are not available
    if not max_attempt:
        print("Opening Default Camera")
        cap = setCamera(index=0)
        if not cap.isOpened():
            print("Default Camera is not Available!")
            exit()
        return cap

# Detect faces in the camera feed
def detectFace(frame):
    # Load the Haar Cascade classifier for face detection
    HaarCascade = cv2.CascadeClassifier(
        cv2.samples.findFile(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    )
    # Detect faces in the image
    faceGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = HaarCascade.detectMultiScale(image=faceGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # Return the coordinates of the detected faces
    if len(faces) > 0:
        print(f"Detected {len(faces)} face(s).")
        return faces
    else:
        print("No faces detected.")
        return None

# Extract face embeddings from the detected faces
def extractFaceEmbeddings(face, frame):
    face = cv2.resize(face, (160, 160))     # Resize the face to 160x160 pixels
    face = face / 255.0                     # Normalize the face
    face = np.expand_dims(face, axis=0)     # Expand dimensions to match model input shape
    signature = model.predict(face)         # Predict the face embeddings
    return signature[0]                     # Return the face embeddings as a 1D array

# Save the face embeddings to a file
def saveFaceEmbeddings(signature, filename="signatures.json"):
    # Check if the file exists, if not create it
    if os.path.exists(filename):
        print(f"File {filename} already exists. Updating the file.")
    else:
        print(f"File {filename} does not exist. Creating a new file.")
        # Create an empty JSON file
        json.dump({}, open(filename, mode="w"), indent=4)

    # Load face signatures from database
    data = json.load(open(filename, mode="r"))    
    # Update the database with the new signature
    data.update({"Freya": signature.tolist()})  # Convert numpy array to list for JSON serialization
    # Save the updated database to the file
    json.dump(data, open(filename, mode="w"), indent=4)

# Render a rectangle around the detected faces
def drawRectangle(x, y, width, height, frame, label="Face Detected", distance=0.0):    
    # Draw a rectangle around the detected face
    frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
    frame = cv2.rectangle(frame, (x,y-40), (x+width, y), (0, 255, 0), -2)
    # Draw a label above the rectangle
    frame = cv2.putText(frame, label + ', ' + str(round(distance, 2)), (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

# Face recognition function
def faceRecognition(x, y, width, height, embeddedFaces, mindistance, frame, filename="signatures.json"):
    # Load the face database
    faceDatabase = json.load(open(filename, mode="r"))
    
    # Initialize lists to store identities and distances
    identities, distances = [], []
    
    # Iterate through the face database
    for identity, signature in faceDatabase.items():
        # Calculate the Euclidean distance between the embedded faces and the stored signatures
        distance = np.linalg.norm(signature - embeddedFaces)
        
        # Append the identity and distance to the lists
        identities.append(identity)
        distances.append(distance)
        
    # Find the index of the minimum distance
    min_index = np.argmin(distances)
    # Get the identity and distance of the closest match
    identity = identities[min_index]
    distance = distances[min_index]
        
    # Print the identity and distance for debugging
    print("Identifing faces...")
    print(f"Identity: {identity}, Distance: {distance}\n")

    # Render the rectangle with the label and distance
    if distance > mindistance:
        print("Failed to recognize the face.")
        drawRectangle(x=x, y=y, width=width, height=height, frame=frame, label="Unknown", isKnown=False, distance=distance)
        # extractFaceEmbeddings(face=frame[y:y+height, x:x+width], frame=frame)
        # saveFaceEmbeddings(signature=embeddedFaces, filename=filename)
    else:
        drawRectangle(x=x, y=y, width=width, height=height, frame=frame, label=identity, isKnown=True, distance=distance)