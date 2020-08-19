import pickle
import os
import time
import cv2
# from mtcnn.mtcnn import MTCNN
from face_rec import FaceClassifier
from datagen import extract_face
import numpy as np

FACE_MODEL_PATH = 'face-recog.pkl'
FACENET_JSON_PATH = 'model/facenet_model.json'
FACENET_WEIGHTS_PATH = 'weights/facenet_keras_weights.h5'

# init face detection class
# detection_model = MTCNN() 
# init face classifier class (for getting face embedding and face classification model)
model = FaceClassifier(FACENET_JSON_PATH, FACENET_WEIGHTS_PATH, False, FACE_MODEL_PATH)
# init camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print('Camera not opened, Bye bud!')
	exit()

while True:
	ret, frame = cap.read()
	W,H,_ = frame.shape
	print(frame.shape)

	start = time.time()

	face, bbox = extract_face(frame)
	faces = np.expand_dims(face, axis=0)
	yhat = model.predict(faces)
	prediction = model.get_class(yhat)
	# print(prediction)
	# faces = classifier.detect_faces(frame)
	# bboxes = [face['box'] for face in faces]
	print('fps: '+ str(1/(time.time()-start)))
	
	# for bbox in bboxes:
	cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (255,0,0), 1)
	frame = cv2.putText(frame, prediction[0], (bbox[0]-10,bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)

	cv2.imshow('op', frame)

	if(cv2.waitKey(27)=='q'):
		cv2.destroyAllWindows()
		break  
