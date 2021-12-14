import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
import numpy as np


class AlexNet:
	@staticmethod
	def build(width, height, depth, classes):
		np.random.seed(1000)
		model = Sequential()
		inputShape = (height, width, depth)
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		#1st Convolutional Layer
		model.add(Conv2D(filters=96, input_shape=inputShape, kernel_size=(11,11), strides=(4,4), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

		#2nd Convolutional Layer
		model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

		#3rd Convolutional Layer
		model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

		#4th Convolutional Layer
		model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))

		#5th Convolutional Layer
		model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

		#Passing it to a Fully Connected layer
		model.add(Flatten())
		# 1st Fully Connected Layer
		model.add(Dense(4096, input_shape=inputShape))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		# Add Dropout to prevent overfitting
		model.add(Dropout(0.5))

		#2nd Fully Connected Layer
		model.add(Dense(4096))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		#Add Dropout
		model.add(Dropout(0.5))

		#3rd Fully Connected Layer
		model.add(Dense(1000))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		#Add Dropout
		model.add(Dropout(0.5))

		#Output Layer
		model.add(Dense(classes))
		model.add(BatchNormalization())
		model.add(Activation('softmax'))

		return model