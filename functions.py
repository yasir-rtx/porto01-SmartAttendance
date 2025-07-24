import os

import cv2
import os
import json
import keras
import numpy as np

# load the facenet128 model
@keras.saving.register_keras_serializable()
def scaling(x, scale=1.0):
	return x * scale
model = keras.saving.load_model("hf://logasja/FaceNet")

# Important Variables
# Signatures / Embedded Faces
SIGNATURES_PATH = "./data/signatures.json"


if not os.path.exists(SIGNATURES_PATH):
    print("file : signatures.json => CREATING...")
    json.dump({}, open(SIGNATURES_PATH, mode="w"), indent=4)
    print("file : signatures.json => CREATED...")
else:
    print("file : signatures.json => OK...")

# configure camera
def setCamera(index: int = 0, width: int = 1280, height: int = 720):
    cap = cv2.VideoCapture(index)
    # Set the camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# access camera
def getCamera(index: int = 1, max_try: int = 5, width: int = 1280, height: int = 720):
    try:
        while max_try:
            cap = setCamera(index, width=width, height=height)

            # return cap if camera available
            if cap.isOpened():
                print(f"Camera-{index} is available")
                return cap
            else:
                print(f"Camera-{index} is not available!")
                index += 1
                max_try -= 1

        # open default camera if no additional camera is available
        if max_try == 0:
            print("Access default camera...")
            cap = setCamera(0, width=width, height=height)
            return cap

    except Exception as e:
        print(f"An error occurred while accessing the camera: {e}")
        exit()


# render rectangle arround detected faces
def drawRectangle(face: list, frame: list, label="Face Detected", distance=0.0):
    x, y, width, height = face[0], face[1], face[2], face[3]

    # Calculate font size based on the width of the rectangle
    # 0.002 is scale factor that determine font size
    font_scale = width * 0.002
    font_size = round(max(0.4, min(0.8, font_scale)), 2)

    # Draw a rectangle around the detected face
    frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
    frame = cv2.rectangle(frame, (x, y - 40), (x + width, y), (0, 255, 0), -2)

    # Draw a label above the rectangle
    frame = cv2.putText(
        frame,
        label + ", " + str(round(distance, 2)),
        (x + 5, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return frame


# face detection
def faceDetection(frame: list, mode: str = "all"):
    # load the Haar Cascade classifier
    HaarCascade = cv2.CascadeClassifier(
        cv2.samples.findFile(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    )

    # convert BGR to grayscale
    faceGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    # faceMaps : numpy array -> store face coordinates
    faceMaps = HaarCascade.detectMultiScale(
        image=faceGray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(128, 128),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # return face coordinates
    if len(faceMaps) == 0:
        # debug
        # print("0 faces detected")

        # return empty array
        return np.array([])
    else:
        # debug
        # print(f"{len(faceMaps)} faces detected")

        # return face maps
        if mode == "single":
            return [faceMaps[0]]  # return only first face coordinates
        if mode == "all":
            return faceMaps


# capture and save face image
def saveFaceImage(label: str, face: list):
    saved_path = os.path.join("./img/", label)

    # check if path exist
    if not os.path.exists(saved_path):
        # print(f"Creating folder for : \"{label}\"")
        os.makedirs(saved_path)
    # else:
    # print(f"Folder for \"{label}\" already exists.")

    # counting the number of files in the directory
    file_counter = len(os.listdir(saved_path)) + 1

    # generate filename and join it with face path
    filename = os.path.join(saved_path, f"{label}_{file_counter}.jpg")

    # save face
    try:
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, face)
        print(f"Face image saved as {filename}")

    except Exception as e:
        print(f"Gagal menyimpan gambar: {e}")

# face embedding
def faceEmbedding(label: str):
    label = label.lower()
    # assign empty faces_data and signatures
    faces_data = []
    signatures = []
    
    # load signatures (embedded faces) file
    signatures_data = json.load(open(SIGNATURES_PATH, mode="r"))
    
    # prepare faces data
    faces_path = f"./img/{label}/"
    for face_file in os.listdir(faces_path):
        face = cv2.imread(os.path.join(faces_path, face_file))  # read face file
        face = cv2.resize(face, (160, 160))                     # resize face image
        face = (face - face.mean()) / face.std()                # normalization
        faces_data.append(face)                               # append to faces_data
        
    # convert faces_data to numpy array
    faces_data = np.array(faces_data)
    print(f"face \"{label}\" {faces_data.shape} : OK....")
    
    # generate face signature
    print(f"\"{label}\" face signature : GENERATING...")
    signature = model.predict(faces_data) # type: ignore
    
    # combine 50 embedding to robust the signature
    signature_mean = np.mean(signature, axis=0)
    signature_median = np.median(signature, axis=0)
    
    # convert all numpy arrays to lists for JSON serialization
    signatures.append(signature.tolist())               # [0] : raw signature
    signatures.append(signature_mean.tolist())          # [1] : mean signature
    signatures.append(signature_median.tolist())        # [2] : median signature

    # update data with new signature
    signatures_data.update(({label: signatures}))
    
    # add new signature to signatures.json
    json.dump(signatures_data, open(SIGNATURES_PATH, mode="w"), indent=4)
    print(f"\"{label}\" face signature : GENERATED...")
    
    # render label and distance
def formatIdentity(label: str, distance: float):
    # format label
    new_label= label[0].upper() + label[1:]
    
    # format distance
    new_distance= round(distance, 4)
    
    return new_label, new_distance

# face recognition
def faceRecognition(face, threshold: float = 7.0):
    # prepare face
    face = cv2.resize(face, (160, 160))         # Resize the face to 160x160 pixels
    face = (face - face.mean()) / face.std()    # Normalize the face
    face = np.expand_dims(face, axis=0)         # Expand dimensions to match model input shape
    face = model.predict(face)                  # type: ignore # generate signature
    
    # assign list for recognition
    recognition = []
    
    # load signatures.json
    signatures = json.load(open(SIGNATURES_PATH, mode="r"))
    
    # calculate distance
    for label, signature in signatures.items():
        distance = np.linalg.norm(signature[2] - face)
        recognition.append([label, distance])
        
    # find identity
    identity = (min(recognition, key=lambda d: d[1]))
    
    # debug
    print("Label : ", formatIdentity(identity[0], identity[1]))
    print(recognition)

    # set threshold
    if identity[1] <= threshold:
        label, dist = identity[0], identity[1]
    else:
        label, dist = "unknown", identity[1]
    
    # return label and distance
    return formatIdentity(label, dist)