import cv2
import numpy as np
import time
from mtcnn.mtcnn import MTCNN
from skimage.exposure import equalize_hist

#init pre-trained model
# classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# fps range (14.24,20.97)
# limit: only frontal face + occlusion problem + classifier needs tuning

classifier = MTCNN()
#fps range (2,3)
# limit: fps low, but highly robust

#init cam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('bam!!')
    exit()
while True:
    #bboxes
    ret, frame = cap.read()
    W,H,_ = frame.shape
    print(frame.shape)
    start = time.time()
    # bboxes = classifier.detectMultiScale(frame, 1.1, 3) # img, scaleFactor, minNeigh
    # ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # channels =cv2.split(ycrcb)
    # cv2.equalizeHist(channels[0], channels[0])
    # cv2.merge(channels, ycrcb)
    # img = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
    # # frame = equalize_hist(frame)
    # frame = cv2.medianBlur(frame, 5)
    faces = classifier.detect_faces(frame)
    bboxes = [face['box'] for face in faces]
    print('fps: '+ str(1/(time.time()-start)))
    for bbox in bboxes:
        cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[0]+bbox[2], bbox[0]+bbox[3]), (255,0,0), 1)

    cv2.imshow('op', frame)

    if(cv2.waitKey(27)=='q'):
        cv2.destroyAllWindows()
        break
