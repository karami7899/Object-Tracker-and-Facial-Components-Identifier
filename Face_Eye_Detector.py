import cv2
import numpy as np


facexml = cv2.CascadeClassifier('face.xml')
eyexml = cv2.CascadeClassifier('eye.xml')
#smilexml = cv2.CascadeClassifier('smile.xml')
#eyexml = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#smilexml = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

cap = cv2.VideoCapture(0)

while(True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facexml.detectMultiScale(gray)
    

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)
        img_crop_gray = gray[x:x+w, y:y+h]
        img_crop_colorful = frame[x:x+w, y:y+h]
        text = "{} , {}".format(x,y)
        cv2.putText(frame, text, org, font, fontScale,
                  color, thickness, cv2.LINE_AA, False)
        
        eyes = eyexml.detectMultiScale(img_crop_gray)
        for (x2, y2, w2, h2) in eyes:
            cv2.rectangle(img_crop_colorful, (x2, y2),
                          (x2+w2, y2+h2), (0, 0, 255), 3)
        

        
    cv2.imshow('out', frame)
    if(cv2.waitKey(1) & 0XFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()








