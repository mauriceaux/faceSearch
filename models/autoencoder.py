from random import shuffle
from os import listdir
from os.path import isfile, join
from escritorConsola.escritorConsola import Escritor
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation
import os

class Autoencoder:

	def setEncoderDim(self, dim):
		self.encoderDim = dim

	def setInputDim(self, dim):
		self.inputSize = dim

	def getEncoder(self):
		self.encoder = Sequential()
		self.encoder.add(Dense(input_dim=self.inputSize, units=self.encoderDim*8))
		self.encoder.add(Activation('tanh'))
		self.encoder.add(Dense(units=self.encoderDim*8))
		self.encoder.add(Activation('tanh'))
		self.encoder.add(Dense(units=self.encoderDim*4))
		self.encoder.add(Activation('tanh'))
		self.encoder.add(Dense(units=self.encoderDim*2))
		self.encoder.add(Activation('tanh'))
		self.encoder.add(Dense(units=self.encoderDim))
		self.encoder.add(Activation('softmax'))
		optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
		self.encoder.compile(loss='mean_absolute_error', optimizer=optimizer)

		try:
			self.encoder.load_weights(os.path.normpath(os.getcwd() + "/models/pesos/encoder.h5"), True)
			print("pesos cargados")
		except OSError:
			print("no se han creado los pesos")

		return self.encoder

	def getDecoder(self):
		self.decoder = Sequential()
		self.decoder.add(Dense(input_dim=self.encoderDim, units=self.encoderDim))
		self.decoder.add(Activation('tanh'))
		self.decoder.add(Dense(units=self.encoderDim*2))
		self.decoder.add(Activation('tanh'))
		self.decoder.add(Dense(units=self.encoderDim*4))
		self.decoder.add(Activation('tanh'))
		self.decoder.add(Dense(units=self.encoderDim*8))
		self.decoder.add(Activation('tanh'))
		self.decoder.add(Dense(units=self.inputSize))
		self.decoder.add(Activation('tanh'))
		optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
		self.decoder.compile(loss='mean_absolute_error', optimizer=optimizer)
		try:
			self.decoder.load_weights(os.path.normpath(os.getcwd() + "/models/pesos/decoder.h5"), True)
			print("pesos cargados")
		except OSError:
			print("no se han creado los pesos")
		return self.decoder
