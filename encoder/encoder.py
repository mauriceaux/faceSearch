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

class Encoder:

	def __init__(self, path="nopath/", batchSize=0):
		self.escritor = Escritor()
		self.path = path
		self.escritor.escribir(str(self) + "_init_", "path es : " + str(self.path))
		self.data = None
		self.encoderDim = None
		self.umbral = 5e-5
		self.pathEncodedData = os.path.normpath(os.getcwd() + "/encoder/encodedData/encoded")
		self.Encoder = None
		self.Decoder = None
		self.Autoencoder = None
		self.GuardarEstado = True
		self.batchSize = batchSize
		self.modelName = "model.h5"
		self.contadorInterno = 0

	def getEncoder(self):
		return self.Encoder

	def getDecoder(self):
		return self.Decoder

	def setPath(self, path):
		self.path = path

	def setModelName(self, modelName):
		self.modelName = modelName

	def setBatchSize(self, batchSize):
		self.batchSize = batchSize

	def setGuardarEstado(self, guardarEstado):
		self.GuardarEstado = guardarEstado

	def setPathEncodedData(self, path):
		self.pathEncodedData = path

	def generar(self):
		self.escritor.escribir(str(self) + " generar", "entrenando")
		

		loss = 1
		while(loss > self.umbral):
			if(self.batchSize > 0):
				self.loadData()
			loss = self.entrenar()
			#loss = 0.0000000000001
			porcentajeCompletado = (100 * self.umbral)/loss
			#self.escritor.escribir(str(self) + " generar", "porcentaje completado: " + str(loss))
			self.escritor.escribir(str(self) + " generar", "porcentaje completado: " + str(porcentajeCompletado))
		print("", end="\n")
		encodedData = self.predecir()
		if(self.batchSize == 0):
			np.save(self.pathEncodedData, encodedData)
		else:
			for i in range(0,encodedData.shape[0]):
				np.save(os.path.normpath(self.pathEncodedData + "/" + str(i)), encodedData[i])
		if self.GuardarEstado == True:
			self.guardarEstado()

		return self.pathEncodedData + ".npy"

	def guardarEstado(self):
		#if self.GuardarEstado:
		self.Encoder.save_weights(os.path.normpath(os.getcwd() + "/encoder/model/encoder-" + self.modelName), True)
		self.Decoder.save_weights(os.path.normpath(os.getcwd() + "/encoder/model/decoder-" + self.modelName), True)
		self.Autoencoder.save_weights(os.path.normpath(os.getcwd() + "/encoder/model/autoencoder-" + self.modelName), True)

	def predecir(self):
		if(self.batchSize == 0):
			if(self.data.all() == None):
				self.data = np.load(os.path.normpath(self.path))
			batch = np.zeros((1, self.data.shape[0]), np.float32)
			batch[0] = self.data[:, 0]
			np.save(self.pathEncodedData, self.Encoder.predict(batch))
			return self.pathEncodedData + ".npy"
		else:
			self.loadData()
			batch = self.data
			batch = batch.reshape(batch.shape[0], batch.shape[1])
			#print("batch.shape", batch.shape)
			#exit()
		return self.Encoder.predict(batch)

	def entrenar(self):
		if self.batchSize == 0:
			batch = np.zeros((1, self.data.shape[0]), np.float32)
			batch[0] = self.data[:, 0]
		else:
			#print(self.data)
			#exit()
			batch = self.data
			#batch = batch.reshape(batch.shape[0], batch.shape[1])
		print(batch)
		return self.Autoencoder.train_on_batch(batch, batch)

	def loadData(self):
		self.data = []
		onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]
		
		
		for i in range(0,self.batchSize):
			if(self.contadorInterno > len(onlyfiles) - 1):
				self.contadorInterno = 0

			data = np.load(os.path.normpath(self.path + "/" + onlyfiles[self.contadorInterno]))
			#print("data.shape " + str(data.shape))
			#exit()
			data = data.reshape(data.shape[1], data.shape[0])
			data = data[0,:]
			#print("self.data.shape " + str(data.shape))
			#exit()
			self.data.append(data)

			self.contadorInterno += 1
		self.data = np.asarray(self.data)


	def initModel(self):
		optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
		#optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
		if(self.batchSize > 0):
			self.data = []
			onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]
			shuffle(onlyfiles)
			for i in range(0,self.batchSize):
				self.data.append(np.load(os.path.normpath(self.path + "/" + onlyfiles[i])))
			self.data = np.asarray(self.data)
			self.data = self.data.reshape(self.data.shape[0],self.data.shape[1])
			self.encoderDim = self.data.shape[1] // 10 

		else:
			self.data = np.load(self.path)
			self.encoderDim = self.data.shape[0] // 10 
		self.Encoder = self.encoder()
		self.Decoder = self.decoder()
		self.Autoencoder = self.autoencoder(self.Encoder, self.Decoder)

		self.Encoder.compile(loss='mean_absolute_error', optimizer=optimizer)
		self.Decoder.compile(loss='mean_absolute_error', optimizer=optimizer)
		self.Autoencoder.compile(loss='mean_absolute_error', optimizer=optimizer)

		try:
			self.Encoder.load_weights(os.path.normpath(os.getcwd() + "/encoder/model/encoder-" + self.modelName), True)
			self.Decoder.load_weights(os.path.normpath(os.getcwd() + "/encoder/model/decoder-" + self.modelName), True)
			self.Autoencoder.load_weights(os.path.normpath(os.getcwd() + "/encoder/model/autoencoder-" + self.modelName), True)
			print("pesos cargados")
		except OSError:
			print("no se han creado los pesos")

	def encoder(self):
	    model = Sequential()
	    if(self.batchSize == 0):
	    	model.add(Dense(input_dim=self.data.shape[0], output_dim=self.encoderDim*8))
	    else:
	    	model.add(Dense(input_dim=self.data.shape[1], output_dim=self.encoderDim*8))
	    model.add(Activation('tanh'))
	    model.add(Dense(output_dim=self.encoderDim*8))
	    model.add(Activation('tanh'))
	    model.add(Dense(output_dim=self.encoderDim*4))
	    model.add(Activation('tanh'))
	    model.add(Dense(output_dim=self.encoderDim*2))
	    model.add(Activation('tanh'))
	    model.add(Dense(output_dim=self.encoderDim))
	    model.add(Activation('softmax'))
	    return model

	def decoder(self):
		model = Sequential()
		model.add(Dense(input_dim=self.encoderDim, output_dim=self.encoderDim))
		model.add(Activation('tanh'))
		model.add(Dense(output_dim=self.encoderDim*2))
		model.add(Activation('tanh'))
		model.add(Dense(output_dim=self.encoderDim*4))
		model.add(Activation('tanh'))
		model.add(Dense(output_dim=self.encoderDim*8))
		model.add(Activation('tanh'))
		if(self.batchSize == 0):
			model.add(Dense(output_dim=self.data.shape[0]))
		else:
			model.add(Dense(output_dim=self.data.shape[1]))
		model.add(Activation('tanh'))
		#model.add(Reshape(self.data.shape, input_shape=self.data.shape[0]))
		return model

	def autoencoder(self, encoder, decoder):
		model = Sequential()
		model.add(encoder)
		model.add(decoder)
		return model
