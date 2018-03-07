import shutil
from random import shuffle
import time
import keras
import numpy as np
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.layers.core import Activation
from keras.optimizers import Adam
from encoder.encoder import Encoder
from escritorConsola.escritorConsola import Escritor
import os
from extrCara.extractor import Extractor

class Clasificador:
	def setEncoderDim(self, dim):
		self.encoderDim = dim

	def setNumClasses(self, num):
		self.numClasses = num

	def getModel(self):
		optimizer = Adam(lr=5e-5, beta_1=0.1, beta_2=0.999)
		self.model = Sequential()
		self.model.add(Dense(input_dim=(self.encoderDim), units=self.encoderDim*2))
		self.model.add(Activation('relu'))
		self.model.add(Dense(units=self.encoderDim*2))
		self.model.add(Activation('relu'))
		self.model.add(Dense(units=self.encoderDim*2))
		self.model.add(Activation('relu'))
		self.model.add(Dense(units=self.numClasses))
		self.model.add(Activation('softmax'))
		self.model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])
		return self.model


