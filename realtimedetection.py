import numpy as np
from keras.models import model_from_json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

# Load the model
json_file = open("Emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade using cv2
import cv2
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

# Function to convert PIL image to OpenCV format
def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Function to convert OpenCV format to PIL image
def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Initialize webcam
webcam = cv2.VideoCapture(0)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

while True:
    i, im = webcam.read()
    if not i:
        break
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), font, 2, (0, 0, 255))
        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass