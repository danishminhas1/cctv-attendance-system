import cv2
import os

# set the path to the folder containing the image files
image_folder = r'C:\Users\Danish\Desktop\NUST 6th Semester\Machine Learning\Recordings\P1L_S4_C1'

# set the frame rate for the video (in fps)
frame_rate = 30

# get the list of image file names in the folder
images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]

# sort the image file names in ascending order
images.sort()

# get the dimensions of the first image file
img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = img.shape

# create a VideoWriter object

video_name = 'output5.avi'
video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))

# loop through the image files, read each image, and write it to the video file
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    video_writer.write(img)

# release the VideoWriter object and destroy all windows
video_writer.release()
cv2.destroyAllWindows()
