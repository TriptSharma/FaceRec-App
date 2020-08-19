import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
from keras.models import model_from_json
import tensorflow as tf
import pickle

MODEL_JSON_PATH = 'model/facenet_model.json'
MODEL_WEIGHTS_PATH = 'weights/facenet_keras_weights.h5'
COMPRESSED_DATASET_PATH = '5-celeb-face.npz'


# GET FACE EMBEDDINGS FROM FACENET MODEL

#Facenet model required normalized input of shape (None, 160, 160, 3)

# def standardise(img):
#     img = img.astype('float32')
#     return tf.image.per_image_standardisation(img)
	

# def preprocess():
#     img_tensor = standardise(img)
#     return tf.expanded_dims(img_tensor, axis=0)

class FaceEmbedding:
	def __init__(self, model_json=MODEL_JSON_PATH, model_weights=MODEL_WEIGHTS_PATH):
		self.model = model_from_json(open(model_json, 'r').read())
		self.model.load_weights(model_weights)
		# print(model.summary())
		# print('Embedding Model loaded successfully!')
		
	def preprocess_input_embedding(self, input_image):
		img = input_image.astype('float32')
		img = (img-img.mean())/img.std()
		return np.expand_dims(img, axis=0)

	def predict_embedding(self, samples):
		return self.model.predict(samples)

	def get_embedding(self, image):
		preprocess_img = self.preprocess_input_embedding(image)
		prediction = self.predict_embedding(preprocess_img)
		return prediction[0]

	def get_batch_embedding(self, batch):
		V = list()
		for img in batch:
			V.append(self.get_embedding(img))
		return V

	def save_embedding(self, filename, *args):
		''' Save face embeddings of a face or a batch in a ocmpressed format (.npz)
			eg. save_embedding(trainx, trainy, valx, valy)''' 
		np.savez_compressed(filename, *args)
		

# model = FaceEmbedding()
# train_embedding_vec = model.get_batch_embedding(trainx)
# val_embedding_vec = model.get_batch_embedding(valx)

# # print(train_embedding_vec)
# model.save_embedding('5-celeb-embedding.npz', train_embedding_vec, trainy, val_embedding_vec, valy)

# CLASSIFICATION BASED ON FACE-EMBEDDINGS
# data = np.load('5-celeb-embedding.npz')
# train_embedding, trainy, val_embedding, valy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class FaceClassifier(FaceEmbedding):
	
	def __init__(self, embed_model_json=MODEL_JSON_PATH, embed_model_weights=MODEL_WEIGHTS_PATH, train=True, *face_classifier_model, **svc_params):
		''' Class to recognise faces.
		embed_model_json = facenet embedding model json file
		embed_model_weights = facenet embedding model weights file
		train = bool value to specify train or inference mode
		face_classifier__model = (optional) path to load face classification model file (containing SVM and LabelEncoder),required if train=False
		svc_params = (optional) keyword arguments for training SVC (according to sklearn.svm.SVC)'''
		super().__init__(embed_model_json, embed_model_weights)
		
		self.in_processor = Normalizer()
		self.train = train
		
		if self.train:
			self.out_processor = LabelEncoder()
			self.classifier = SVC()
			if len(svc_params)!=0:
				self.classifier.set_params(svc_params)
			print('Model Initialized')
		else:
			# get inference model
			if len(face_classifier_model)==0:
				print('path to load face classification model file needed')
				exit()
			with open(face_classifier_model[0], "rb") as f:
				try:
					self.classifier = pickle.load(f)
					self.out_processor = pickle.load(f)
				except Exception:
					print('Incomplete model. SVM and LabelEncider models needed')
					exit()

			# print(data)
			# if len(data)!=2:
			print('Classification Model Loaded Successfully!')

	def preprocess_input(self, xinput):
		xinput_pp = self.in_processor.transform(xinput)
		return xinput_pp

	def preprocess_output(self, yinput):
		yinput_pp = self.out_processor.fit_transform(yinput)
		return yinput_pp

	def fit_classifier(self, trainx, trainy):
		self.classifier.fit(trainx, trainy)
		return self.classifier

	def predict_classifier(self, X):
		return self.classifier.predict(X)


	def accuracy(self, yhat, y):
		return accuracy_score(yhat, y)

	def fit(self, trainx, trainy):
		# get embed - save embed - load embed - preprocess i/o - fit classifier - predict - accuracy
		assert self.train==True
		train_vec = self.get_batch_embedding(trainx)
		# print(train_vec)
		train = self.preprocess_input(train_vec)
		labels = self.preprocess_output(trainy)
		self.fit_classifier(train, labels)
	
	def predict(self, X):
		X_vec = self.get_batch_embedding(X)
		vec = self.preprocess_input(X_vec)
		return self.predict_classifier(vec)

	def save_model(self, filename):
		with open(filename, "wb") as f:
			pickle.dump(self.classifier, f)
			pickle.dump(self.out_processor, f)
		print('Model Saved Successfully!')
		
	def get_class(self, label):
		return self.out_processor.inverse_transform(label)

if __name__ == '__main__':
	
	data = np.load(COMPRESSED_DATASET_PATH)
	trainx, trainy, valx, valy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
	print(len(trainx))
	print('Dataset Loaded Successfully!')

	facerec = FaceClassifier(MODEL_JSON_PATH,MODEL_WEIGHTS_PATH)
	facerec.fit(trainx, trainy)
	facerec.save_model('face-recog.pkl')

	inference = FaceClassifier(MODEL_JSON_PATH,MODEL_WEIGHTS_PATH, False, 'face-recog.pkl')
	yhat = inference.predict(valx)
	train_yhat = inference.predict(trainx)

	# print(yhat)
	train_score = inference.accuracy(train_yhat, trainy)
	val_score = inference.accuracy(yhat, valy)
	print('Accuracy: train:{}, val:{}'.format(train_score*100, val_score*100))
