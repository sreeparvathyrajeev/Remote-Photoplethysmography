import cv2
import numpy as np
import os
import time

# Create a directory to save cropped faces if it doesn't exist
output_dir = 'cropped_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture from the C270 HD WEBCAM
cap = cv2.VideoCapture(2)

# Initialize a counter for cropped face filenames
face_count = 0

# FPS control
target_fps = 30
frame_duration = 1 / target_fps

# Define the target size for the cropped faces
output_size = (256, 256)

while True:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Resize the cropped face to the target size
        resized_face = cv2.resize(face_roi, output_size)

        # Save the resized face to the output directory
        face_filename = os.path.join(output_dir, f"face_{face_count}.jpg")
        cv2.imwrite(face_filename, resized_face)
        face_count += 1

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # --- FPS Limiter ---
    processing_time = time.time() - start_time
    wait_time = int(max(1, (frame_duration - processing_time) * 1000))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()