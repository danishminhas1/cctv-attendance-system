import cv2 as cv
import os
from people import people
from collections import Counter
import numpy as np

features = list()
labels = list()
f = np.load('features_leaving.npy', allow_pickle=True)
l = np.load('labels_leaving.npy', allow_pickle=True)
# DIR = r'C:\Users\Danish\Desktop\NUST 6th Semester\Machine Learning\P1E\P1E_S2_C1'
DIR = r'C:\Users\Danish\Desktop\NUST 6th Semester\Machine Learning\Recordings\P1L_S1_C1'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for person in people:
        #count = 0
        path = os.path.join(DIR, person)
        label = str(person)[2:]
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x: x+w]
                features.append(faces_roi)
                labels.append(label)
            #print(img_path)
            #print(label)
            # count += 1
            # if (count == 30):
            #     break 
    counted_list = Counter(labels)
    
    for item, count in counted_list.items():
        print(f"{item}: {count}")
    print("Length of features list: " + str(len(features)) + " along with labels: " + str(len(labels)))
    #print(labels)
create_train()
print("Training complete")
features = np.array(features, dtype = 'object')
labels = np.array(labels, dtype=int)
features = np.append(features, f, axis = 0)
labels = np.append(labels, l, axis = 0)
print(len(features))
print(len(labels))
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
face_recognizer.save('face_trained_leaving.yml')
np.save('features_leaving.npy', features)
np.save('labels_leaving.npy', labels)

