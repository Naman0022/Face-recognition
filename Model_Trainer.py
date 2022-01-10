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


# import face_recognition
# import cv2
# import numpy as np
# import os
# import glob

# faces_encodings = []
# faces_names = []
# cur_direc = os.getcwd()
# path = os.path.join(cur_direc, 'Face recognition\\data\\faces\\')
# list_of_files = [f for f in glob.glob(path+'*.jpg')]
# number_files = len(list_of_files)
# names = list_of_files.copy()

# for i in range(number_files):
#     globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
#     globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
#     faces_encodings.append(globals()['image_encoding_{}'.format(i)])
# # Create array of known names
#     names[i] = names[i].replace(cur_direc, "")  
#     faces_names.append(names[i])

# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True

# video_capture = cv2.VideoCapture(0)
# while True:
#     ret, frame = video_capture.read()
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = small_frame[:, :, ::-1]
#     if process_this_frame:
#         face_locations = face_recognition.face_locations( rgb_small_frame)
#         face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
#         face_names = []
#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces (faces_encodings, face_encoding)
#             name = "Unknown"
#             face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = faces_names[best_match_index]
#             face_names.append(name)
#     # process_this_frame = not process_this_frame
# # face_locations,
# # Display the results
#     for (top, right, bottom, left), name in face_names:
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4
# # Draw a rectangle around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
# # Input text label with a name below the face
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         print(name)
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
# # Display the resulting image
#     cv2.imshow('Video', frame)
# # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     process_this_frame = not process_this_frame
# # process_this_frame = not process_this_frame

# # __________________________________________________________________________________________________________
# # from PIL import Image
# # import imagehash
# # hash0 = imagehash.average_hash(Image.open('D:\\Python\\Face recognition\\img\\testimage (1).jpg')) 

# # cutoff = 6

# # from os import walk

# # f = []
# # for (dirpath, dirnames, filenames) in walk('D:\\Python\\Face recognition\\img'):
# #     f.extend(filenames)
# #     print(f)
# #     break
# # i=1
# # while i <= len(f):
# #     name = 'testimage ('+str(i)+').jpg'
# #     path = 'D:\\Python\\Face recognition\\img\\'+ name
# #     hash1 = imagehash.average_hash(Image.open(path))

# #     if hash0 - hash1  < cutoff:
# #         print(name+' images are similar')
# #     else:
# #         print(name+' images are not similar')
# #     i=i+1
# # _____________________________________________________________________________________________