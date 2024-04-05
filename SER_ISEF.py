#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from keras.models import load_model

# Parameters
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
MODEL_PATH = 'Emotion_Voice_Detection_Model.h5'  # Update this to your model path
VOICE_THRESHOLD = 0.01  # Define a threshold for voice detection

# Load the pre-trained model
model = load_model(MODEL_PATH)

while True:
    def record_audio():
        print("Recording...")
        audio = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Finished recording.")
        # Save the recorded data as a WAV file
        write(WAVE_OUTPUT_FILENAME, RATE, audio)
        return audio

    def check_voice_presence(audio, threshold=VOICE_THRESHOLD):
        """Check if the recorded audio has voice present based on energy threshold."""
        energy = np.sqrt(np.mean(audio**2))
        return energy > threshold

    def preprocess_audio(file_path, model_input_shape=(216, 1)):
        # Load audio file
        y, sr = librosa.load(file_path, sr=RATE, mono=True)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=model_input_shape[1])

        if mfccs.shape[1] < model_input_shape[0]:
            pad_width = model_input_shape[0] - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mfccs.shape[1] > model_input_shape[0]:
            mfccs = mfccs[:, :model_input_shape[0]]

        mfccs = mfccs.T.reshape(-1, model_input_shape[0], model_input_shape[1])
        return mfccs

    def predict_emotion(audio_features):
        predictions = model.predict(audio_features)
        emotion_index = np.argmax(predictions)
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predicted_emotion = emotions[emotion_index] if emotion_index < len(emotions) else "Unknown"
        return predicted_emotion

    def main():
        # Record and save audio
        audio = record_audio()
        # Check if voice is present
        if not check_voice_presence(audio):
            print("\nNo voice detected.")
            return

        # Preprocess the audio
        audio_features = preprocess_audio(WAVE_OUTPUT_FILENAME)
        # Predict emotion
        emotion = predict_emotion(audio_features)
        file = open('ser_txt.txt', 'w')
        file.write(emotion)
        print(f"Predicted Emotion: {emotion}")

    if __name__ == "__main__":
        main()


# In[ ]:




