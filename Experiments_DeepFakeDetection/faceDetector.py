import time

import cv2
import dlib

filepath='./'   #the picture`s path
filepath2='./'  #the 81 points detectors path

img=cv2.imread(filepath)
height,width,channels=img.shape
rescale=0.5
grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faceDector=dlib.get_frontal_face_detector()
landmarkPredcitor=dlib.shape_predictor(filepath2+"shape_predictor_81_face_landmarks.dat")

faces=faceDector(grayImg,1)
time_start=time.time()
for face in faces:
    landmarks=landmarkPredcitor(img,face)
    for pt in landmarks.parts():
        pt_pos=(pt.x,pt.y)
        cv2.circle(img,pt_pos,2,(0,255,0),1)

time_end=time.time()
img=cv2.resize(img,(int(width*rescale),int(height*rescale)))

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

