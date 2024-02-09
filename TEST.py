#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip uninstall serial --yes')


# In[ ]:





# In[3]:


get_ipython().system('pip install pyserial')


# In[2]:


# CNN Final

import cv2
import numpy as np
from keras.models import load_model

def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    f1 = 2 * tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-10)
    return f1

# Load the trained emotion recognition model. Set the path to your model file.
emotion_model = load_model('model_optimal.h5', custom_objects={"f1_metric": f1_metric})

# Load the OpenCV haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a video frame
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = gray_frame[y:y+h, x:x+w]

        # Resize the face region to match the model's input size
        face = cv2.resize(face, (48, 48))

        # Normalize the face image
        face = face / 255.0

        # Make a prediction by passing the preprocessed face to the emotion recognition model
        emotion_prediction = emotion_model.predict(np.expand_dims(face, axis=0))

        # Get the emotion label based on the predicted class
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        emotion_label = emotions[np.argmax(emotion_prediction)]

        # Draw a rectangle around the detected face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with face detection and emotion recognition
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[1]:


# Xception Final

import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    f1 = 2 * tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-10)
    return f1

# Load the trained emotion recognition model. Set the path to your model file.
emotion_model = load_model('best_model_Xception.h5', custom_objects={"f1_metric": f1_metric})
tf.keras.utils.register_keras_serializable("f1_metric")(f1_metric)

# Load the OpenCV haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a video frame
    if not ret:
        break

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Resize the face region to match the model's input size (299x299 for Xception)
        face = cv2.resize(face, (224, 224))

        # Convert to RGB color format (Xception requires RGB)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize the face image
        face = face / 255.0

        # Make a prediction by passing the preprocessed face to the emotion recognition model
        emotion_prediction = emotion_model.predict(np.expand_dims(face, axis=0))

        # Get the emotion label based on the predicted class
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        emotion_label = emotions[np.argmax(emotion_prediction)]

        # Draw a rectangle around the detected face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with face detection and emotion recognition
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[1]:


# Pre-trained model: AffectNET

import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf

class TestModels:
    def __init__(self, h5_address: str, GPU=True):
        self.exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        self.model = self.load_model(h5_address=h5_address)

    def load_model(self, h5_address: str):
        model = tf.keras.models.load_model(h5_address, custom_objects={'tf': tf})
        return model

    def recognize_fer(self, frame):
        # Resize the frame to match the model's input size (224x224 for your model)
        frame = resize(frame, (224, 224, 3))

        # Expand dimensions to create a batch of size 1
        frame = np.expand_dims(frame, axis=0)

        # Perform emotion recognition
        prediction = self.model.predict_on_batch([frame])
        exp = np.array(prediction[0])

        # Get the recognized emotion
        emotion_label = self.exps[np.argmax(exp)]

        return emotion_label

# Initialize the TestModels class
tester = TestModels(h5_address='AffectNet_6336.h5')

# Load the OpenCV haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a video frame
    if not ret:
        break

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Use the recognize_fer method to get the emotion label
        emotion_label = tester.recognize_fer(face)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the emotion label at the top of the bounding box
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        

    # Display the frame with face detection
    cv2.imshow("Emotion Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[6]:


get_ipython().system('pip install tensorflow==2.13.0rc0')


# In[3]:


# Voice + FACE
import cv2
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import time

class TestModels:
    def __init__(self, h5_address: str, GPU=True):
        self.exps = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        self.model = self.load_model(h5_address=h5_address)

    def load_model(self, h5_address: str):
        model = tf.keras.models.load_model(h5_address, custom_objects={'tf': tf})
        return model

    def recognize_fer(self, frame):
        # Resize the frame to match the model's input size (224x224 for your model)
        frame = resize(frame, (224, 224, 3))

        # Expand dimensions to create a batch of size 1
        frame = np.expand_dims(frame, axis=0)

        # Perform emotion recognition
        prediction = self.model.predict_on_batch([frame])
        exp = np.array(prediction[0])

        # Get the recognized emotion
        emotion_label = self.exps[np.argmax(exp)]

        return emotion_label

# Initialize the TestModels class
tester = TestModels(h5_address='AffectNet_6336.h5')

# Load the OpenCV haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the recognizer for voice sentiment analysis
analyzer = SentimentIntensityAnalyzer()
recognizer = sr.Recognizer()

while True:
    time.sleep(3)
    
    # Read a video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Use the recognize_fer method to get the emotion label for the face
        emotion_label_face = tester.recognize_fer(face)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the face emotion label at the top of the bounding box
        cv2.putText(frame, emotion_label_face, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Use the recognizer for voice sentiment analysis
        with sr.Microphone() as source:
            print("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            try:
                audio = recognizer.listen(source, timeout=5)  # Increase the timeout to 5 seconds (adjust as needed)
                text = recognizer.recognize_google(audio, language="ko-KR")
                print(f"You said: {text}")

                sentiment_scores = analyzer.polarity_scores(text)
                # Determine sentiment based on the compound score
                if sentiment_scores['compound'] >= 0.05:
                    sentiment = "Positive"
                elif sentiment_scores['compound'] <= -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                print("Sentiment:", sentiment)

                # Display the voice sentiment label next to the face bounding box
                cv2.putText(frame, f"Voice Sentiment: {sentiment}", (x + w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except sr.UnknownValueError:
                print("Sorry, I couldn't understand the audio.")
            except sr.RequestError as e:
                print(f"Sorry, an error occurred while processing the audio: {e}")

    # Display the frame with face detection
    cv2.imshow("Emotion Recognition", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    f1 = 2 * tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-10)
    return f1

rgb = [0,0,0]

# Load the trained emotion recognition model. Set the path to your model file.
emotion_model = load_model('best_model_Xception.h5', custom_objects={"f1_metric": f1_metric})
tf.keras.utils.register_keras_serializable("f1_metric")(f1_metric)

# Load the OpenCV haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a video frame
    if not ret:
        break

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Resize the face region to match the model's input size (299x299 for Xception)
        face = cv2.resize(face, (224, 224))

        # Convert to RGB color format (Xception requires RGB)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize the face image
        face = face / 255.0

        # Make a prediction by passing the preprocessed face to the emotion recognition model
        emotion_prediction = emotion_model.predict(np.expand_dims(face, axis=0))

        # Get the emotion label based on the predicted class
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        emotion_label = emotions[np.argmax(emotion_prediction)]

        file = open('txt.txt', 'w')
        if np.argmax(emotion_prediction) == 0:
            file.write('1')
            rgb = [255, 0, 0] #Red
        elif np.argmax(emotion_prediction) == 1:
            file.write('2')
            rgb = [255, 68, 51] #Orange
        elif np.argmax(emotion_prediction) == 2:
            file.write('3')
            rgb = [102, 2, 60] #Violet
        elif np.argmax(emotion_prediction) == 3:
            file.write('4')
            rgb = [0, 255, 0] #Green
        elif np.argmax(emotion_prediction) == 4:
            file.write('5')
            rgb = [0, 0, 0] #White
        elif np.argmax(emotion_prediction) == 5:
            file.write('6')
            rgb = [0, 0, 255] #Blue
        else: 
            file.write('7')
            rgb = [255, 255, 0] #Yellow

        # Draw a rectangle around the detected face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (rgb[0], rgb[1], rgb[2]), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (rgb[0], rgb[1], rgb[2]), 2)

    # Display the frame with face detection and emotion recognition
    cv2.imshow("Emotion Recognition", frame)

    # Escape
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[20]:


py_serial.write(3)
time.sleep(3)
py_serial.write(1)


# In[8]:


import speech_recognition as sr
with sr.Microphone() as source:
    recognizer = sr.Recognizer()
    print("듣는 중... 이제 말해주세요!")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source, timeout=3)
    print(f"오디오 기간: {len(audio.frame_data) / audio.sample_width / audio.sample_rate} 초")


# In[ ]:


import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    f1 = 2 * tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-10)
    return f1

rgb = [0,0,0]

# Load the trained emotion recognition model. Set the path to your model file.
emotion_model = load_model('cnn/best_model_Xception-2.h5', custom_objects={"f1_metric": f1_metric})
tf.keras.utils.register_keras_serializable("f1_metric")(f1_metric)

# Load the OpenCV haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a video frame
    if not ret:
        break

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Resize the face region to match the model's input size (299x299 for Xception)
        face = cv2.resize(face, (224, 224))

        # Convert to RGB color format (Xception requires RGB)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize the face image
        face = face / 255.0

        # Make a prediction by passing the preprocessed face to the emotion recognition model
        emotion_prediction = emotion_model.predict(np.expand_dims(face, axis=0))

        # Get the emotion label based on the predicted class
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        emotion_label = emotions[np.argmax(emotion_prediction)]

        file = open('txt.txt', 'w')
        if np.argmax(emotion_prediction) == 0:
            file.write('1')
            rgb = [255, 0, 0] #Red
        elif np.argmax(emotion_prediction) == 1:
            file.write('2')
            rgb = [255, 68, 51] #Orange
        elif np.argmax(emotion_prediction) == 2:
            file.write('3')
            rgb = [102, 2, 60] #Violet
        elif np.argmax(emotion_prediction) == 3:
            file.write('4')
            rgb = [0, 255, 0] #Green
        elif np.argmax(emotion_prediction) == 4:
            file.write('5')
            rgb = [0, 0, 0] #White
        elif np.argmax(emotion_prediction) == 5:
            file.write('6')
            rgb = [0, 0, 255] #Blue
        else: 
            file.write('7')
            rgb = [255, 255, 0] #Yellow

        # Draw a rectangle around the detected face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (rgb[0], rgb[1], rgb[2]), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (rgb[0], rgb[1], rgb[2]), 2)

    # Display the frame with face detection and emotion recognition
    cv2.imshow("Emotion Recognition", frame)

    # Escape
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


import time
import serial

py_serial = serial.Serial(
     port = 'COM9',
     baudrate = 9600,
)

while True:
     time.sleep(4)
     f = open('txt.txt', 'r')
     commend = f.read()
     py_serial.write(commend.encode())
     print(commend)

