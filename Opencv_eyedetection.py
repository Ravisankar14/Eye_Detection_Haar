import numpy as np
import cv2



#Loding face detection xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Loding eye detection xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Selecting the video source
cap = cv2.VideoCapture(0)

#Getting height and width of the video frame
height=int(cap.get(3))
width=int(cap.get(4))

#Setting the fourcc format
fourcc=cv2.VideoWriter_fourcc(*'MP4V')

#Creating the object for VideoWriter
out=cv2.VideoWriter("output.mp4",fourcc,20.0,(height,width))

#Creating the empty window
cv2.namedWindow("img",cv2.WINDOW_NORMAL)

#Creating loop for processing frames
while(True):
    #Reading one frame
    ret, frame = cap.read()

    #Converting the RGB frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecting the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Creating loop for prcessing each face
    for (x,y,w,h) in faces:

        #Drawing the rectangle for the face
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #Croping only the face(gray & color)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #Detecting the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        num_eyes=len(eyes)

        #Checking face detected or not
        print (num_eyes)
        if num_eyes > 1:
                frame = cv2.putText(frame, 'Eye Detected',(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(0,255,0),2,cv2.LINE_AA)
        
        #Creating loop for prcessing each face
        for (ex,ey,ew,eh) in eyes:

            #Drawing the rectangle for the eye
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #Writing the frames into video
    out.write(frame)

    #Showing the output in the window 
    cv2.imshow('img',frame)

    #Creating waitKey for keyboard response and Quiting the window 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        #Breaking the while loop after pressing the Esc
        break

#Releasing the capture object
cap.release()

#Releasing the writer object
out.release()

#Destroying all the windows
cv2.destroyAllWindows()
