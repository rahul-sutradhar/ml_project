import numpy as np
   from keras.models import model_from_json
   from PIL import Image, ImageDraw, ImageFont
   import imageio
   import requests
   from io import BytesIO
   import streamlit as st
   import tensorflow as tf

   # Check if GPU is available
   print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

   # Load the model
   json_file = open("Emotiondetector.json", "r")
   model_json = json_file.read()
   json_file.close()
   model = model_from_json(model_json)
   model.load_weights("emotiondetector.h5")

   # Ensure the model uses the GPU
   with tf.device('/GPU:0'):
       model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

   # Function to extract features
   def extract_features(image):
       feature = np.array(image)
       feature = feature.reshape(1, 48, 48, 1)
       return feature / 255.0

   # Function to get a frame from the video stream
   def get_frame(url):
       response = requests.get(url)
       img = Image.open(BytesIO(response.content))
       return img

   # URL of the phone camera stream
   stream_url = 'http://192.168.1.2:8080/video'

   labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

   # Streamlit UI
   st.title("Real-Time Emotion Detection")

   if st.button("Start Detection"):
       while True:
           im = get_frame(stream_url)
           gray = im.convert('L')

           # Detect faces (Replace this with actual face detection code)
           # For example, using dlib: faces = face_detector(gray)
           faces = []  # Placeholder for detected faces

           draw = ImageDraw.Draw(im)
           try:
               for (p, q, r, s) in faces:
                   image = gray.crop((p, q, p + r, q + s))
                   draw.rectangle([p, q, p + r, q + s], outline=(255, 0, 0), width=2)
                   image = image.resize((48, 48))
                   img = extract_features(image)
                   pred = model.predict(img)
                   prediction_label = labels[pred.argmax()]

                   # Draw text on the image
                   draw.text((p - 10, q - 10), prediction_label, fill=(0, 0, 255))

               st.image(im, caption='Processed Image')
           except Exception as e:
               st.error(f"An error occurred: {e}")