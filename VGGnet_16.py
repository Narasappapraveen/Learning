import tensorflow as tf
from tensorflow.contrib.layers import flatten,xavier_initializer
import numpy as np
import random



class VGGnet:
	def __init__(self,x,n_classes,keep_prob_conv,keep_prob,mu=0,sigma=0.1,):
		self.x_dict		   = x
		self.n_out 		   = n_classes
		self.mu    		   = mu
		self.sigma 		   = sigma
		self.Keep_prob_conv = keep_prob_conv
		self.Keep_prob     = keep_prob
	def get_model(self):
		x_input = self.x_dict
		# Layer1
		conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = self.mu, stddev = self.sigma),name='Conv1_W')
		conv1_b = tf.Variable(tf.zeros(32),name='Conv1_b')
		conv1   = tf.nn.conv2d(x_input,conv1_W, strides=[1, 1, 1, 1], padding='SAME',name='Conv1') + conv1_b
		
		# ReLu Activation.
		conv1 = tf.nn.relu(conv1,name='Relu1')
		
		# Layer 2 (Convolutional): Input = 32x32x32. Output = 32x32x32.
		conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = self.mu, stddev = self.sigma),name='Conv2_W')
		conv2_b = tf.Variable(tf.zeros(32),name='Conv2_b')
		conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME',name='Conv2') + conv2_b
		
		# ReLu Activation.
		conv2 = tf.nn.relu(conv2,name='Relu2')
		
		# Layer 3 (Pooling): Input = 32x32x32. Output = 16x16x32.
		conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',name='Max_pool1')
		conv2 = tf.nn.dropout(conv2, self.Keep_prob_conv,name='Droput1')
		
		# Layer 4 (Convolutional): Input = 16x16x32. Output = 16x16x64.
		conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = self.mu, stddev = self.sigma),name='Conv3_W')
		conv3_b = tf.Variable(tf.zeros(64),name='Conv3_b')
		conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME',name='Conv3') + conv3_b
		
		# ReLu Activation.
		conv3 = tf.nn.relu(conv3,name='Relu3')
		
		# Layer 5 (Convolutional): Input = 16x16x64. Output = 16x16x64.
		conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = self.mu, stddev = self.sigma),name='Conv4_W')
		conv4_b = tf.Variable(tf.zeros(64),name='Conv4_b')
		conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME',name='Conv4') + conv4_b
		
		# ReLu Activation.
		conv4 = tf.nn.relu(conv4,name='relu4')
		
		# Layer 6 (Pooling): Input = 16x16x64. Output = 8x8x64.
		conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',name='Max_pool2')
		conv4 = tf.nn.dropout(conv4, self.Keep_prob_conv,name='Droput12') # dropout
		
		# Layer 7 (Convolutional): Input = 8x8x64. Output = 8x8x128.
		conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = self.mu, stddev = self.sigma),name='Conv5_W')
		conv5_b = tf.Variable(tf.zeros(128),name='Conv5_b')
		conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME',name='Conv5') + conv5_b
		
		# ReLu Activation.
		conv5 = tf.nn.relu(conv5,name='Relu5')
		
		# Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
		conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = self.mu, stddev = self.sigma),name='Conv6_W')
		conv6_b = tf.Variable(tf.zeros(128),name='Conv6_b')
		conv6   = tf.nn.conv2d(conv5, conv6_W, strides=[1, 1, 1, 1], padding='SAME',name='Conv6') + conv6_b
		
		# ReLu Activation.
		conv6 = tf.nn.relu(conv6,name='Relu6')
		
		# Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
		conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',name='Maxpool3')
		conv6 = tf.nn.dropout(conv6, self.Keep_prob_conv,name='Dropout3') # dropout

		# Flatten. Input = 4x4x128. Output = 2048.
		fc0   = flatten(conv6)

		# Layer 10 (Fully Connected): Input = 2048. Output = 128.
		fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean = self.mu, stddev = self.sigma),name='Fc1_W')
		fc1_b = tf.Variable(tf.zeros(128),name='Fc1_b')
		fc1   = tf.matmul(fc0, fc1_W ,name='Fc1') + fc1_b

		# ReLu Activation.
		fc1    = tf.nn.relu(fc1,name='Relu7')
		fc1    = tf.nn.dropout(fc1, self.Keep_prob,name='Dropout4') # dropout
		
		# Layer 11 (Fully Connected): Input = 128. Output = 128.
		fc2_W  = tf.Variable(tf.truncated_normal(shape=(128, 128), mean = self.mu, stddev = self.sigma),name='Fc2_W')
		fc2_b  = tf.Variable(tf.zeros(128),name='Fc2_b')
		fc2    = tf.matmul(fc1, fc2_W,name='Fc2') + fc2_b

		# ReLu Activation.
		fc2    = tf.nn.relu(fc2,name='Relu8')
		fc2    = tf.nn.dropout(fc2, self.Keep_prob,name='Dropout4') # dropout

		# Layer 12 (Fully Connected): Input = 128. Output = n_out.
		fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, self.n_out), mean = self.mu, stddev = self.sigma),name='Fc3_W')
		fc3_b  = tf.Variable(tf.zeros(self.n_out),name='Fc3_b')
		logits = tf.matmul(fc2, fc3_W,name='Fc3') + fc3_b
		return logits
		
		
		
		
		
	

class VGGnet_16:
	def __init__(self,x,n_classes,keep_prob_conv,keep_prob,mu=0,sigma=0.1,):
		self.x_dict		   = x
		self.n_out 		   = n_classes
		self.mu    		   = mu
		self.sigma 		   = sigma
		self.Keep_prob_conv = keep_prob_conv
		self.Keep_prob     = keep_prob
			
		
	def conv_layer(self,input_tensor,kw, kh, n_out, name, dw=1, dh=1,activation_fn=tf.nn.relu ,padding = 'SAME'):
		n_in = input_tensor.get_shape()[-1].value
		with tf.variable_scope(name):
			weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
			biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
			conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding = padding)
			activation = activation_fn(tf.nn.bias_add(conv, biases))
			return activation
	
	def fully_connected(self,input_tensor, n_out, name,activation_fn=tf.nn.relu):
		n_in = input_tensor.get_shape()[-1].value
		with tf.variable_scope(name):
			weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
			biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
			logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
			return activation_fn(logits)
	
	def pool(self,input_tensor, kh, kw, dh, dw , name):
		return tf.nn.max_pool(input_tensor,
							  ksize=[1, kh, kw, 1],
							  strides=[1, dh, dw, 1],
							  padding='VALID',name=name)
	def get_model(self):
		x_input = self.x_dict
		
		# Layer1		
		conv1 = self.conv_layer(x_input,3, 3, 32, name='conv1')
		
		# Layer 2 (Convolutional): Input = 32x32x32. Output = 32x32x32.
		conv2 = self.conv_layer(conv1,3, 3, 32, name='conv2')
		
		# Layer 3 (Pooling): Input = 32x32x32. Output = 16x16x32.		
		conv2 = self.pool(conv2,2, 2, 2, 2, name='Max_pool1')
		conv2 = tf.nn.dropout(conv2, self.Keep_prob_conv,name='Droput1')
		
		# Layer 4 (Convolutional): Input = 16x16x32. Output = 16x16x64.
		conv3 = self.conv_layer(conv2,3, 3, 64, name='conv3')
		
		# Layer 5 (Convolutional): Input = 16x16x64. Output = 16x16x64.
		conv4 = self.conv_layer(conv3,3, 3, 64, name='conv4')		
		
		# Layer 6 (Pooling): Input = 16x16x64. Output = 8x8x64.
		conv4 = self.pool(conv4,2, 2, 2, 2, name='Max_pool2')
		conv4 = tf.nn.dropout(conv4, self.Keep_prob_conv,name='Droput2') # dropout
		
		# Layer 7 (Convolutional): Input = 8x8x64. Output = 8x8x128.
		conv5 = self.conv_layer(conv4,3, 3, 128, name='conv5')
		
		# Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
		conv6 = self.conv_layer(conv5,3, 3, 128, name='conv6')
		
		# Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
		conv6 = self.pool(conv6,2, 2, 2, 2, name='Max_pool3')
		conv6 = tf.nn.dropout(conv6, self.Keep_prob_conv,name='Dropout3') # dropout

		# Flatten. Input = 4x4x128. Output = 2048.
		fc0   = flatten(conv6)

		# Layer 10 (Fully Connected): Input = 2048. Output = 128.
		fc1	   = self.fully_connected(fc0,128,name = 'Fc1')
		fc1    = tf.nn.dropout(fc1, self.Keep_prob,name='Dropout4') # dropout
		
		# Layer 11 (Fully Connected): Input = 128. Output = 128.
		fc2	   = self.fully_connected(fc1,128,name = 'Fc2')
		fc2    = tf.nn.dropout(fc2, self.Keep_prob,name='Dropout4') # dropout

		# Layer 12 (Fully Connected): Input = 128. Output = n_out.
		logits	   = self.fully_connected(fc2, self.n_out ,name = 'Fc3')
		return logits		
		
		
	def get_TL_model(Input):

		# Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
		Conv1_NL = self.conv_layer(Input,3, 3, 128, name='Conv1_NL')
		
		# Layer 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
		Conv1_NL = self.pool(Conv1_NL,2, 2, 2, 2, name='max_pool1_NL')
		Conv1_NL = tf.nn.dropout(Conv1_NL, self.Keep_prob_conv,name='Dropout1_NL') # dropout

		# Flatten. Input = 4x4x128. Output = 2048.
		fc0_NL   = flatten(Conv1_NL)

		# Layer 10 (Fully Connected): Input = 2048. Output = 128.
		Fc1_NL = self.fully_connected(fc0_NL,128,name = 'Fc1_NL')
		Fc1_NL = tf.nn.dropout(Fc1_NL, self.Keep_prob,name='Dropout2_NL') # dropout
		
		# Layer 11 (Fully Connected): Input = 128. Output = 128.
		Fc2_NL	   = self.fully_connected(Fc1_NL,128,name = 'Fc2_NL')
		Fc2_NL    = tf.nn.dropout(Fc2_NL, self.Keep_prob,name='Dropout3_NL') # dropout

		# Layer 12 (Fully Connected): Input = 128. Output = n_out.
		logits_NL	   = self.fully_connected(Fc2_NL, self.n_out ,name = 'Fc3_NL')
		return logits_NL		
		
	
		
		
		
		
		
		
		
		
		
		
		
