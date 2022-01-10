import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#find path of xml file containing haarcascade file 
cascPathface = 'D:\Python\Face recognition\Cascades\Haarcascade _frontalFace_alt2.xml'
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier('D:\Python\Face recognition\Cascades\Haarcascade _frontalFace_alt2.xml')
# load the known faces and embeddings saved in last file

data = pickle.loads(open('face_enc', "rb").read())
print(faceCascade.load('D:\Python\Face recognition\Cascades\Haarcascade _frontalFace_alt2.xml'))
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
attendence=[]
index=1
flag=0
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.05,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches1 = face_recognition.face_distance(data["encodings"],encoding)
        matches = face_recognition.api.compare_faces(data["encodings"],encoding,tolerance=0.45)
        
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            if name =="Unknown":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,20,20), 1)
            
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (20,255,20), 0)
            strength = w*0.005
            lv=2
            if strength<1.7:
                lv=1
            
            cv2.putText(frame, name, (x +25, y+h+20), cv2.FONT_HERSHEY_TRIPLEX,strength, (225, 225, 225), lv)
            # print(name ,' in ',attendence,' - ',type(name in attendence),name in attendence)
            if (name in attendence) is False:
                if name != "Unknown":
                    attendence.append(name)
            
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(30) & 0xff
    if k==27 or k == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
index=0
while index<len(attendence):
    print(attendence[index],"\tP")
    index=index+1