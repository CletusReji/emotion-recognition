# EmoVision : Real-Time Emotion Detection Using CNNs

## Overview

EmoVision is a real-time emotion detection system that uses deep learning techniques to identify human emotions with reasonable accuracy from facial expressions.

EmoVision uses Computer Vision (OpenCV and MediaPipe) and a Convolutional Neural Network (CNN) trained on the FER2013 dataset to classify emotions from facial expressions.

The system works on live webcam input and shows a bounding box around the detected face with the predominant emotion (calculated over a sliding window for stability), along with the confidence percentage. Users can also capture snapshots with predicted labels.

The core focus of our application is to showcase the capabilities of such a system in recognizing emotions. Through the use of Convolutional Neural Network (CNN) trained on Facial Emotion Recognition (FER) dataset, with over 35000 labeled emotions across 6 emotion classes that include: Angry, Fear, Happy, Sad, Surprise and Neutral. Our model processes live camera feed as input and predicts the emotion in real time, that can enable it’s application in various fields like mental health monitoring, user  experience analysis, human-computer interaction and intuitive gaming.

EmoVision aims to bridge the emotional gap between humans and machines by leveraging computer vision and deep learning, paving the way for more empathetic and responsive systems.


## Supported Emotions

* Happy
* Sad
* Surprise
* Neutral
* Anger
* Fear


## Features

* Real-time face capture and detection using OpenCV and MediaPipe
* CNN trained on the standard FER dataset
* Emotion label displayed along with bounding box of the recognised face and confidence percentage
* Sliding window is used for stable predictions
* Option to click pictures of the displayed output with labels


## Model and Training

* [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)
* Architechture : Custom trained CNN
* Training
  - Input : Grayscale (48x48 images)
  - Optimizer : Adam 
  - Loss : Categorical Cross Entropy
  - Metrics : Accuracy


## Demo

### Welcome Page

![welc](https://github.com/user-attachments/assets/7b09ed53-065e-478b-8373-741faca3dc47)

### Home Page

![home](https://github.com/user-attachments/assets/1079e455-8bc4-45e4-9e71-1a9a8ff6b9a3)


## Contributors

- [Adarsh Mohan P](https://github.com/Adarshmohanp)
- [Cletus Reji C](https://github.com/CletusReji)
- [Leanne Roslyn Biju](https://github.com/leannebiju)
