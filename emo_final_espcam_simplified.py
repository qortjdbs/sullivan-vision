import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import time
import requests

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

#ESP-CAM
URL = "http://192.168.57.180"
AWB = True

# Initialize the webcam
cap = cv2.VideoCapture(URL + ":81/stream")

TT = 0

while True:
    #time.sleep(4)
    ret, frame = cap.read()  # Read a video frame
    if not ret:
        break

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    Trig = 0

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Resize the face region to match the model's input size (299x299 for Xception)
        face = cv2.resize(face, (224, 224))

        # Convert to RGB color format (Xception requires RGB)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize the face image
        face = face / 255.0

        start = time.time()

        # Make a prediction by passing the preprocessed face to the emotion recognition model
        emotion_prediction = emotion_model.predict(np.expand_dims(face, axis=0))

        # Get the emotion label based on the predicted class
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        emotion_label = emotions[np.argmax(emotion_prediction)]

        file = open('fer_txt.txt', 'w')
        if np.argmax(emotion_prediction) == 0:
            file.write('1')
            rgb = [255, 0, 0] #Red
        elif np.argmax(emotion_prediction) == 1:
            file.write('1')
            rgb = [255, 68, 51] #Orange
        elif np.argmax(emotion_prediction) == 2:
            file.write('1')
            rgb = [102, 2, 60] #Violet
        elif np.argmax(emotion_prediction) == 3:
            file.write('4')
            rgb = [0, 255, 0] #Green
        elif np.argmax(emotion_prediction) == 4:
            file.write('5')
            rgb = [0, 0, 0] #White
        elif np.argmax(emotion_prediction) == 5:
            file.write('1')
            rgb = [0, 0, 255] #Blue
        else: 
            file.write('4')
            rgb = [255, 255, 0] #Yellow

        end = time.time()
        
        if(TT==1):
            print(end-start)
        TT+=1

        # Draw a rectangle around the detected face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (rgb[0], rgb[1], rgb[2]), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (rgb[0], rgb[1], rgb[2]), 2)

        time.sleep(1)

    # Display the frame with face detection and emotion recognition
    cv2.imshow("Emotion Recognition", frame)

    #time.sleep(3)

    # Escape
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()