import os
import cv2

# Define directory to store collected images
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Create data directory if it doesn't exist

# Set the number of classes and dataset size per class
number_of_classes = 3
dataset_size = 100

# Initialize video capture (camera index may vary based on device)
cap = cv2.VideoCapture(2)

# Loop to create folders and collect data for each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)  # Create a folder for each class if not exists

    print('Collecting data for class {}'.format(j))

    # Initial prompt to prepare user to start data collection
    while True:
        ret, frame = cap.read()  # Read frame from camera
        # Display prompt on the frame
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        # Wait for 'q' key to start collecting images for this class
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect specified number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Read frame from camera
        cv2.imshow('frame', frame)
        cv2.waitKey(25)  # Display frame briefly

        # Save current frame as an image in the respective class directory
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1  # Increment image counter

# Release video capture and close OpenCV windows after collection
cap.release()
cv2.destroyAllWindows()