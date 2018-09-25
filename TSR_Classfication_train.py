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

from VGGnet_16 import VGGnet,VGGnet_16

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

	

x = tf.placeholder(tf.float32,(None,32,32,1),name='Input')
y = tf.placeholder(tf.int32,(None),name='Ground_truth')

keep_prob = tf.placeholder(tf.float32 ,name='keep_prob')       # For fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32 ,name='keep_prob_conv')  # For convolutional layers

# Training set preprocessing
normalized_images = preprocess(x_train)

EPOCHS = 30
BATCH_SIZE = 200
learning_rate=0.001
DIR = 'C:/Praveen/TSR/traffic_sign_classification_german-master/Saved_Models/VGGNet16/'
Log_dir = 'C:/Praveen/TSR/traffic_sign_classification_german-master/VGG16_logs/'

model_name = "VGGNet" 
#VGGNet_Model = VGGnet(x,n_classes = n_classes,keep_prob_conv = keep_prob_conv,keep_prob = keep_prob)
VGGNet_Model = VGGnet_16(x,n_classes = n_classes,keep_prob_conv = keep_prob_conv,keep_prob = keep_prob)
logits = VGGNet_Model.get_model()
one_hot_y = tf.one_hot(y, n_classes)
with tf.variable_scope('Cost_function'):		
	# Training operation
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels=one_hot_y )
	loss_operation = tf.reduce_mean(cross_entropy, name='loss')
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	training_operation = optimizer.minimize(loss_operation)
with tf.variable_scope('Accuracy'):	
	# Accuracy operation
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

# Validation set preprocessing
x_valid_preprocessed = preprocess(x_valid)


saver = tf.train.Saver(tf.global_variables())

prev_accuracy = 0
 

with tf.Session() as sess:

	writer_train = tf.summary.FileWriter((os.path.join(Log_dir, 'train')), sess.graph)
	writer_test  = tf.summary.FileWriter((os.path.join(Log_dir, 'test')), sess.graph)
	
	sess.run(tf.global_variables_initializer())
	num_examples = len(y_train)
	print("Training...")
	print()
	
	for i in range(EPOCHS):
		train_accuracy= 0
		train_loss    = 0
		normalized_images, y_train = shuffle(normalized_images, y_train)
		for offset in range(0, num_examples, BATCH_SIZE):
			end = offset + BATCH_SIZE
			batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]
			_,training_loss, training_accuracy = sess.run([training_operation, loss_operation , accuracy_operation],feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})
			train_accuracy += (training_accuracy * len(batch_x) )
			train_loss	   += (training_loss * len(batch_x) )
		#pdb.set_trace() 
		train_accuracy = train_accuracy/ num_examples
		train_loss = train_loss / num_examples
		print(train_accuracy)
		print(train_loss)
		# Training summary
		Train_acc = tf.Summary(value=[ tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy)])
		Train_loss = tf.Summary(value=[ tf.Summary.Value(tag="train_loss", simple_value=train_loss)])
		#train_summary = tf.summary.merge([Train_acc,Train_loss])
		#writer_train.add_summary( train_summary, global_step= i)	
		writer_train.add_summary( Train_acc, global_step= i)	
		writer_train.add_summary( Train_loss, global_step= i)	
		writer_train.flush()
		# validation summary
		valid_loss,valid_accuracy = sess.run([loss_operation,accuracy_operation],feed_dict={x: x_valid_preprocessed, y: y_valid, keep_prob: 1.0, keep_prob_conv: 1.0 })
		Vald_acc = tf.Summary(value=[ tf.Summary.Value(tag="valid_accuracy", simple_value=valid_accuracy)])
		Vald_loss = tf.Summary(value=[ tf.Summary.Value(tag="valid_loss", simple_value=valid_loss)])
		#val_loss = tf.summary.scalar('Val_loss', valid_loss)
		#val_acc  = tf.summary.scalar('Val_accuracy', valid_accuracy)
		#Valid_summary = tf.summary.merge([val_acc,val_loss])
		writer_test.add_summary(Vald_acc, global_step=i)
		writer_test.add_summary(Vald_loss, global_step=i)		
		writer_test.flush()
		
		print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (valid_accuracy*100)))
		print('		Validation Loss = %f' % valid_loss)
		if valid_accuracy > prev_accuracy:
			saver.save(sess = sess, save_path = os.path.join(DIR, model_name))
			prev_accuracy = valid_accuracy
	print("Model saved")








