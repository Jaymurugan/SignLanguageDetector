import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize Mediapipe hands module for static image processing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing collected image data
DATA_DIR = './data'

# Initialize lists to store processed hand landmarks data and corresponding labels
data = []
labels = []

# Iterate over each class directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list for processed coordinates
        x_ = []  # List to store x-coordinates for normalization
        y_ = []  # List to store y-coordinates for normalization

        # Load image and convert to RGB for Mediapipe processing
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to extract hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Loop over landmarks to normalize based on minimum x and y values
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize and store landmark data for the hand in data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalized x
                    data_aux.append(y - min(y_))  # Normalized y

            data.append(data_aux)  # Append processed data for current image
            labels.append(dir_)  # Label the data with the directory name (class)

# Save processed data and labels to a pickle file for training
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
