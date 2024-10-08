# Emotion Recognition Device for the Visually Impaired

This project presents a wearable device designed to assist visually impaired individuals in perceiving emotions during conversations. The device recognizes facial expressions and voice tones using deep learning algorithms and transmits this information to the user through vibrations. It is developed in the form of glasses, making it portable and hands-free.

## Key Features:
- **Facial Expression Recognition (FER)**: Recognizes facial emotions using an ESP32 camera and a pre-trained Xception model, achieving 98.55% accuracy.
- **Speech Emotion Recognition (SER)**: Recognizes emotions from voice input via Conv1D and LSTM models, achieving 97.25% accuracy.
- **Wireless Communication**: Uses Bluetooth Low Energy (BLE) for wireless transmission between the emotion recognition system and the vibration sensor.
- **Vibration Feedback**: Delivers immediate feedback on emotions through distinct vibration patterns.
- **Mobile Application**: Provides a simple interface for controlling the device, supporting emotion recognition in three modes (positive, negative, and neutral).

## Hardware:
- **ESP32 Board**: Processes and transmits emotional data.
- **Vibration Sensor**: Provides haptic feedback to convey emotions.
- **Battery-powered**: Lightweight and portable with a rechargeable lithium-ion battery.

## Usage:
The device is intended to improve communication for visually impaired individuals by offering real-time emotional feedback through both facial expression and speech emotion recognition. It is ideal for everyday use due to its wireless, hands-free design.
