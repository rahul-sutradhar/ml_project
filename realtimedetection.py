import numpy as np
from keras.models import model_from_json
from PIL import Image, ImageDraw, ImageFont
import imageio
import imageio_ffmpeg as iio_ffmpeg

# Load the model
json_file = open("Emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar Cascade using an alternative method (e.g., pre-trained model)
# We will assume the face detection model is loaded and ready to use
# For example, using a pre-trained model from dlib or another library
# face_detector = some_face_detection_model

# Function to extract features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature / 255.0

# Initialize webcam
webcam = imageio.get_reader('<video0>')

labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

for im in webcam:
    # Convert image to grayscale
    gray = Image.fromarray(im).convert('L')
    
    # Detect faces (Replace this with actual face detection code)
    # For example, using dlib: faces = face_detector(gray)
    faces = []  # Placeholder for detected faces

    try:
        for (p, q, r, s) in faces:
            image = gray.crop((p, q, p+r, q+s))
            draw = ImageDraw.Draw(im)
            draw.rectangle([p, q, p+r, q+s], outline=(255, 0, 0), width=2)
            image = image.resize((48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Draw text on the image
            draw.text((p-10, q-10), prediction_label, fill=(0, 0, 255))

        im.show()
    except Exception as e:
        print(f"An error occurred: {e}")