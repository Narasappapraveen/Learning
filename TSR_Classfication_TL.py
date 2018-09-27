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
from tensorflow.contrib.layers import flatten,conv2d
from sklearn.metrics import confusion_matrix
import pdb

from VGGnet_16 import VGGnet , VGGnet_16

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
	
def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if adam_opt.get_slot(var, name) is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.variables_initializer(adam_vars)

# Training set preprocessing
normalized_images = preprocess(x_train)
x_valid_preprocessed = preprocess(x_valid)

model_name = "VGGNet" 
EPOCHS = 30
BATCH_SIZE = 200
learning_rate=0.001
DIR	    = 'C:/Praveen/TSR/traffic_sign_classification_german-master/Saved_Models/VGGNet16_TL/'
Log_dir = 'C:/Praveen/TSR/traffic_sign_classification_german-master/VGGNet16_TL_logs/'

tf.reset_default_graph()
new_graph = tf.Graph()
# Test set preprocessing
#x_test_preprocessed = preprocess(x_test)
with tf.Session(graph=new_graph) as sess:
    # saver = tf.train.import_meta_graph('../TSR/traffic_sign_classification_german-master/Saved_Models/VGGNet.meta')
    # saver.restore(sess,tf.train.latest_checkpoint('../TSR/traffic_sign_classification_german-master/Saved_Models'))	

    saver = tf.train.import_meta_graph('C:/Praveen/TSR/traffic_sign_classification_german-master/Saved_Models/VGGNet16/VGGNet.meta')
    saver.restore(sess,tf.train.latest_checkpoint('C:/Praveen/TSR/traffic_sign_classification_german-master/Saved_Models/VGGNet16'))	

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("Input:0")
    y = graph.get_tensor_by_name("Ground_truth:0")
    keep_prob_conv = graph.get_tensor_by_name("keep_prob_conv:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    #loss_operation = graph.get_tensor_by_name("Cost_function/loss:0")
    #Acc_operation = graph.get_tensor_by_name("Accuracy/accuracy:0")
    VGGNet_Model = VGGnet_16(x,n_classes = n_classes,keep_prob_conv = keep_prob_conv,keep_prob = keep_prob)
    #Restore the network till 5th layer
    conv5_out = graph.get_tensor_by_name("conv5/Relu:0")
    conv5_out = tf.stop_gradient(conv5_out)

    #Adding own layers
    with tf.variable_scope("trainable_section"):
        # Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
        Conv1_NL = tf.layers.conv2d(inputs=conv5_out,filters=128,kernel_size=[3, 3],padding="same",activation=tf.nn.relu, name='Conv1_NL')    

        # Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
        Conv1_NL = tf.nn.max_pool(Conv1_NL,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name='max_pool1_NL')
        Conv1_NL = tf.nn.dropout(Conv1_NL, keep_prob_conv,name='Dropout1_NL') # dropout
        # Flatten. Input = 4x4x128. Output = 2048.
        fc0_NL   = flatten(Conv1_NL)
            # Layer 10 (Fully Connected): Input = 2048. Output = 128.
        Fc1_NL = tf.layers.dense(fc0_NL,256,name = 'Fc1_NL')
        Fc1_NL = tf.nn.dropout(Fc1_NL, keep_prob,name='Dropout2_NL') # dropout

        # Layer 11 (Fully Connected): Input = 128. Output = 128.
        Fc2_NL = tf.layers.dense(Fc1_NL,512,name = 'Fc2_NL')
        Fc2_NL = tf.nn.dropout(Fc2_NL, keep_prob ,name='Dropout3_NL') # dropout

        # Layer 12 (Fully Connected): Input = 128. Output = n_out.
        logits_NL  = tf.layers.dense(Fc2_NL, n_classes ,name = 'Fc3_NL')

    # get the variables declared in the scope "trainable_section", training only newley added layers
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "trainable_section")

    one_hot_y = tf.one_hot(y, n_classes)
    with tf.variable_scope('Cost_function1'):		
        # Training operation
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits =logits_NL, labels=one_hot_y )
        loss_operation = tf.reduce_mean(cross_entropy, name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        training_operation = optimizer.minimize(loss_operation, var_list=trainable_vars )
    with tf.variable_scope('Accuracy1'):	
        # Accuracy operation
        correct_prediction = tf.equal(tf.argmax(logits_NL, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
		
    #pdb.set_trace()
    saver = tf.train.Saver()
    prev_accuracy = 0

    #Summary writers
    writer_train = tf.summary.FileWriter((os.path.join(Log_dir, 'train')), sess.graph)
    writer_test  = tf.summary.FileWriter((os.path.join(Log_dir, 'test')), sess.graph)

    trainable_variable_initializers = [var.initializer for var in trainable_vars]
    sess.run(trainable_variable_initializers)
    reset_opt_vars = adam_variables_initializer(optimizer, trainable_vars)
    #momentum_initializers = [var for var in tf.global_variables() if 'Momentum' in var.name]
    #var_list_init = tf.variables_initializer(momentum_initializers)
    sess.run(reset_opt_vars)
    #sess.run(tf.variables_initializer(optimizer.variables()))
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
        print("EPOCH {} : Training Accuracy = {:.3f}%".format(i+1, (train_accuracy*100)))
        print('		Training Loss = %f' % train_loss)
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

        print("     Validation Accuracy = {:.3f}%".format(valid_accuracy*100))
        print('		Validation Loss = %f' % valid_loss)
        if valid_accuracy > prev_accuracy:
            saver.save(sess = sess, save_path = os.path.join(DIR, model_name))
            prev_accuracy = valid_accuracy
    print("Model saved")
   

	
	
	
	
