import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix
import pdb

from VGGnet_16 import VGGnet

#Loading the data
training_file   = 'C:/Praveen/TSR/traffic_sign_classification_german-master/train.p'
validation_file = 'C:/Praveen/TSR/traffic_sign_classification_german-master/valid.p'
testing_file    = 'C:/Praveen/TSR/traffic_sign_classification_german-master/test.p'

with open(training_file , mode='rb') as f:
	train = pickle.load(f)
with open(validation_file , mode='rb') as f:
	valid = pickle.load(f)
with open(testing_file , mode='rb') as f:
	test  = pickle.load(f)

# Mapping ClassID to Traffic sign names	
signs = []
with open('C:/Praveen/TSR/traffic_sign_classification_german-master/signnames.csv' , 'r') as csvfile:
	signnames = csv.reader(csvfile, delimiter=',')
	next(signnames,None)
	for row in signnames:
		signs.append(row[1])
	csvfile.close()

x_train , y_train = train['features'] , train['labels']
x_valid , y_valid = valid['features'] , valid['labels']
x_test  , y_test  = test['features']  , test['labels']

# Number of examples in each category 
n_train , n_valid , n_test = x_train.shape[0] , x_valid.shape[0] , x_test.shape[0] 

#No of Classes in dataset
n_classes = len(np.unique(y_train))

# Ploting the sample train, valid, test images
def list_images(dataset , dataset_y , ylabel='', cmap = None) :
	
	plt.figure(figsize=(15,16))
	for i in range(6):
		plt.subplot(1,6,i+1)
		indx = random.randint(0, len(dataset))
		cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
		plt.imshow(dataset[indx],cmap=cmap)
		plt.xlabel(signs[dataset_y[indx]])
		plt.ylabel(ylabel)
		plt.xticks([])
		plt.yticks([])
	plt.tight_layout(pad=0, h_pad=0, w_pad=0)
	plt.show()

#list_images(x_train,y_train,'Training_Examples')
#list_images(x_valid,y_valid,'Validation_Examples')
#list_images(x_test,y_test,'Testing_Examples')




# Histogrma function
def histogram_plot(dataset, label):
	hist, bins = np.histogram(dataset, bins=n_classes)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.xlabel(label)
	plt.ylabel("Image count")
	plt.show()
	
# Plotting histograms of the count of each sign

#histogram_plot(y_train, "Training examples")
#histogram_plot(y_valid, "Validation examples")
#histogram_plot(y_test, "Testing examples")

# Shuffling : To increase randomness
x_train , y_train = shuffle(x_train , y_train)

def gray_scale(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Local Histogram equalization
def local_histo_equalize(image):
	kernel = morp.disk(30)
	img_local = rank.equalize(image , selem=kernel)
	return img_local


# Normalization
def image_normalize(image):
	image = np.divide(image,255)
	return image


display_preprocess_image_samples = 0
if display_preprocess_image_samples is 1:	
	# Sample images after grayscaling
	gray_images = list(map(gray_scale, x_train))
	list_images(gray_images, y_train, "Gray Scale image", "gray")
	# Sample images after Local Histogram Equalization
	equalized_images = list(map(local_histo_equalize, gray_images))
	list_images(equalized_images, y_train, "Equalized Image", "gray")
	# Sample images after normalization
	n_training = x_train.shape
	normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
	for i, img in enumerate(equalized_images):
		normalized_images[i] = image_normalize(img)
	list_images(normalized_images, y_train, "Normalized Image", "gray")
	pdb.set_trace()
	normalized_images = normalized_images[..., None]

# All preprocessing steps in a single step
def preprocess(data):
	gray_images = list(map(gray_scale , data))
	equalized_images = list(map(local_histo_equalize , gray_images))
	n_training = data.shape
	normalized_images = np.zeros((n_training[0],n_training[1],n_training[2]))
	for i, img in enumerate(equalized_images):
		normalized_images[i] = image_normalize(img)
	normalized_images = normalized_images[..., None]
	return normalized_images

	

# Training set preprocessing
#normalized_images = preprocess(x_train)




def y_predict( X_data, BATCH_SIZE=64):
	num_examples = len(X_data)
	y_pred = np.zeros(num_examples, dtype=np.int32)
	sess = tf.get_default_session()
	for offset in range(0, num_examples, BATCH_SIZE):
		batch_x = X_data[offset:offset+BATCH_SIZE]
		y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(logits, 1),feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})
	return y_pred

def evaluate( X_data, y_data, BATCH_SIZE=64):
	num_examples = len(X_data)
	total_accuracy = 0
	sess = tf.get_default_session()
	for offset in range(0, num_examples, BATCH_SIZE):
		batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
		loss,accuracy = sess.run([loss_operation,accuracy_operation],feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })
		total_accuracy += (accuracy * len(batch_x))
		loss += (loss * len(batch_x))
	return loss/ num_examples ,total_accuracy / num_examples



print('Test Case started')

# Test set preprocessing
x_test_preprocessed = preprocess(x_test)

with tf.Session() as sess:
	# saver = tf.train.import_meta_graph('../TSR/traffic_sign_classification_german-master/Saved_Models/VGGNet.meta')
	# saver.restore(sess,tf.train.latest_checkpoint('../TSR/traffic_sign_classification_german-master/Saved_Models'))	
	
	saver = tf.train.import_meta_graph('C:/Praveen/TSR/traffic_sign_classification_german-master/Saved_Models/VGGNet.meta')
	saver.restore(sess,tf.train.latest_checkpoint('C:/Praveen/TSR/traffic_sign_classification_german-master/Saved_Models'))	
	
	graph = tf.get_default_graph()
	
	x = graph.get_tensor_by_name("Input:0")
	y = graph.get_tensor_by_name("Ground_truth:0")
	pdb.set_trace()
	keep_prob_conv = graph.get_tensor_by_name("keep_prob_conv:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")
	loss_operation = graph.get_tensor_by_name("Cost_function/loss:0")
	Acc_operation = graph.get_tensor_by_name("Accuracy/accuracy:0")

	
	test_loss,test_accuracy = sess.run([loss_operation,Acc_operation],feed_dict={x: x_test_preprocessed, y: y_test, keep_prob: 1.0, keep_prob_conv: 1.0 })
	print("Test Accuracy = {:.1f}%".format(test_accuracy*100))
	print('Test Loss = %f' % test_loss)

	
	














