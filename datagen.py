import cv2
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import os
import numpy as np
# import tensorflow as tf
detector = MTCNN()

def extract_face(image, required_size=(160, 160)):
	'''extract a single face from a given image'''
	# create the detector, using default weights
	# detect faces in the image
	results = detector.detect_faces(image)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = image[y1:y2, x1:x2]
	# resize pixels to the model size
	face = cv2.resize(face,required_size)
	print(image.shape, face.shape)
	# face_array = asarray(image)
	return face, [x1,y1,x2,y2]

def extract_face_from_file(filename, required_size=(160, 160)):
	# load image from file
	pixels = cv2.imread(filename)
	pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
	image, bbox = extract_face(pixels, required_size)
	
	return image, bbox

# specify folder to plot
trainfolder = 'data/train/'
valfolder = 'data/val/'
def show_faces(folder):
	# enumerate files
	i = 1
	for filename in os.listdir(folder):
		# path
		path = folder + filename
		# path = tf.convert_to_tensor(path, dtype=tf.string)
		# get face
		print(path)
		face, _ = extract_face(path)
		print(i, face.shape)
		# plot
		pyplot.subplot(2, 7, i)
		pyplot.axis('off')
		pyplot.imshow(face)
		i += 1
	pyplot.show()

def get_db(root_folder):
	faces_db= []
	names_db = []
	count = 0
	for root, _, file in os.walk(root_folder, topdown=True):
		names_db.extend([root.split('/')[-1] for x in range(len(file))])
		# print(names_db)
		for f in file:
			img_f = os.path.join(root, f)
			face, _ = extract_face_from_file(img_f)
			faces_db.append(face)
			count+=1
	# print(len(faces_db), len(names_db), count)
	return np.asarray(faces_db), np.array(names_db)

def generate_face_dataset(trainfolder=trainfolder, validationfolder=valfolder, outputfile='5-celeb-face.npz'):
	trainx, trainy = get_db(trainfolder)
	valx, valy = get_db(validationfolder)

	np.savez_compressed(outputfile, trainx, trainy, valx, valy)

if __name__ == '__main__':
	# generate_face_dataset()
	show_faces(trainfolder+'tript_sharma/')
	print('Face Dataset Generation Successful!')