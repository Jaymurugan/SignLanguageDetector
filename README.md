# Hand Gesture Classification Project

## Project Overview
This project focuses on capturing hand images to build a dataset, processing them to train a hand gesture classifier, and performing real-time inference on live video frames. The key steps include image collection, dataset creation, model training, and real-time gesture recognition.

## Files

- **`collect_imgs.py`**: Captures and stores images for each gesture class.
- **`create_dataset.py`**: Processes the captured images by detecting hand landmarks, normalizing coordinates, and saving them with labels for training purposes.
- **`train_classifier.py`**: Allows for image capturing with options to specify the number of classes and dataset size, facilitating the training of the classifier.
- **`inference_classifier.py`**: Loads the trained model and classifies hand gestures in real-time video frames.

## Instructions

1. **Image Collection**: Run `collect_imgs.py` to capture hand images for each gesture class.
2. **Dataset Creation**: Execute `create_dataset.py` to process images and prepare training data.
3. **Model Training**: Train the classifier using the processed data with `train_classifier.py`.
4. **Real-time Inference**: Use `inference_classifier.py` to predict hand gestures in real-time video frames.

## Requirements

- `mediapipe`
- `cv2`
- `pickle`
- `numpy`
