from imutils import paths
import face_recognition
import pickle
import cv2
import os
import threading
 
#get paths of each file in folder named Images
#Images here contains my data(folders of various persons)
imagePaths = list(paths.list_images('D:\\Python\\Face recognition\\faces'))
knownEncodings = []
knownNames = []

# loop over the image paths
print("Encoding Images...")

for (i, imagePath) in enumerate(imagePaths):
    
    
    # if __name__ == "__main__":
    #     t1 = threading.Thread(target=task1, name='t1')
    #     t2 = threading.Thread(target=task2, name='t2')
    # extract the person name from the image path
    print('\n',i)
    name = imagePath.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    print(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb,model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)


i=0
while i<len(knownNames):
    if knownNames[i] != knownNames[i-1]:
        print('Found: '+knownNames[i])
    i=i+1
#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()
