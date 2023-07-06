from people import people
import os
import shutil

DIR = r'C:\Users\Danish\Desktop\NUST 6th Semester\Machine Learning\P1L_S4_C1'
p = list()
for person in people:
    path = os.path.join(DIR, person)
    for filename in os.listdir(path):
        if filename.startswith('.'):
            file_path = os.path.join(path, filename)
            os.remove(file_path)

def remove_empty_directories(path):
    # Iterate over all files and directories in the given path
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            # Join the current directory path with the current directory name
            dir_path = os.path.join(root, name)
            # If the directory is empty, remove it using shutil.rmtree()
            if not os.listdir(dir_path):
                shutil.rmtree(dir_path)

# Call the function to remove all empty directories recursively from the given directory
remove_empty_directories(DIR)

