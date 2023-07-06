import cv2
import numpy as np
from datetime import datetime

# Load the pre-trained Haar cascade classifier
face_cascade = cv2.CascadeClassifier('haar_face.xml')

# Load the fine-tuned model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_trained_leaving.yml')

# Set the minimum confidence level and durability threshold
MIN_CONFIDENCE = 60
DURABILITY_THRESHOLD = 2.5
distance_threshold = 7000

# Initialize variables for storing the current recognized face, its confidence, and its durability
current_face = None
current_confidence = None
current_durability = 0

# Open the video capture device
cap = cv2.VideoCapture('output5.avi')
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# roi = (0, frame_height//1.5, frame_width, frame_height//1.5)
attendance = list()
# Loop through the frames in the video
while True:
    # Read the next frame from the video
    ret, frame = cap.read()

    # If the frame was not successfully read, break out of the loop
    if not ret:
        break
    # x, y, w, h = roi
    # roi_frame = frame[y:y+h, x:x+w]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the Haar cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        if w * h > distance_threshold and w * h < 12000:
        # Extract the face region of interest (ROI) from the grayscale frame
            face_roi = gray[y:y+h, x:x+w]

            # Recognize the face using the fine-tuned model
            label, confidence = model.predict(face_roi)

            # If the confidence level is above the minimum threshold and the recognized face is the same as the current face
            if confidence >= MIN_CONFIDENCE and label == current_face:
                # Increment the current face's durability
                current_durability += 1

                # If the current face's durability has exceeded the threshold, update the output and reset the durability
                if current_durability >= DURABILITY_THRESHOLD:
                    #print('Recognized face:', label, confidence)
                    attendance.append([label, datetime.now().strftime("%H:%M:%S")])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    current_durability = 0
            # If the confidence level is above the minimum threshold but the recognized face is different from the current face
            elif confidence >= MIN_CONFIDENCE:
                # Update the current face, confidence, and durability
                current_face = label
                current_confidence = confidence
                current_durability = 1
            # If the confidence level is below the minimum threshold, reset the current face, confidence, and durability
            else:
                current_face = None
                current_confidence = None
                current_durability = 0

    # Display the output frame
    cv2.imshow('frame', frame)

    # If the user presses the 'q' key, break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(attendance)
# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()


