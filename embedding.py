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
        print('Model loaded successfully!')
        
    def preprocess_input(self, input_image):
        img = input_image.astype('float32')
        img = (img-img.mean())/img.std()
        return np.expand_dims(img, axis=0)

    def predict(self, samples):
        return self.model.predict(samples)

    def get_embedding(self, image):
        preprocess_img = self.preprocess_input(image)
        prediction = self.predict(preprocess_img)
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
        
# data = np.load(COMPRESSED_DATASET_PATH)
# trainx, trainy, valx, valy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
# print(len(trainx))
# print('Dataset Loaded Successfully!')

# model = FaceEmbedding()
# train_embedding_vec = model.get_batch_embedding(trainx)
# val_embedding_vec = model.get_batch_embedding(valx)

# # print(train_embedding_vec)
# model.save_embedding('5-celeb-embedding.npz', train_embedding_vec, trainy, val_embedding_vec, valy)

# CLASSIFICATION BASED ON FACE-EMBEDDINGS
data = np.load('5-celeb-embedding.npz')
train_embedding, trainy, val_embedding, valy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

from sklearn.preprocessing import Normalizer, LabelEncoder

    
def preprocess_input(xinput):
    processor = Normalizer()
    xinput_pp = processor.transform(xinput)
    return xinput_pp

def preprocess_output(yinput):
    processor = LabelEncoder()
    yinput_pp = processor.fit_transform(yinput)
    return yinput_pp

train = preprocess_input(train_embedding)
val = preprocess_input(val_embedding)

label_train = preprocess_output(trainy)
label_val = preprocess_output(valy)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(train, label_train)

yhat_train = classifier.predict(train)
yhat = classifier.predict(val)

from sklearn.metrics import accuracy_score
train_score = accuracy_score(yhat_train, label_train)
val_score = accuracy_score(yhat, label_val)

print('Accuracy: train:{}, val:{}'.format(train_score*100, val_score*100))

FACE_MODEL_PATH = 'face_model.pkl'
pickle.dump(classifier, open(FACE_MODEL_PATH, 'wb'))