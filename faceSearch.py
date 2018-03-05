#from models.clasificador  import Clasificador
from keras.layers.normalization import BatchNormalization
#from models.autoencoder import Autoencoder
from data.loader import DataLoader
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from data.loader import DataLoader
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
#from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten

import os
import shlex, subprocess
import shutil
import cv2
import numpy as np


class faceSearch:

	def __init__(self):
		print("init")
		#self.clasificador = Clasificador()
		#self.autoencoder = Autoencoder()
		self.pathLib = os.path.normpath(os.getcwd() + "/lib/deepfakes/faceswap.py")
		self.pathImgGenerada = os.path.normpath(os.getcwd() + "/tmp/faces")
		self.modeloIniciado = False
		self.dataLoader = DataLoader()
		self.dataLoader.setPathClassData(os.path.normpath(os.getcwd() + "/bd/categoriasImg"))
		self.dataLoader.setPathTrainingData(os.path.normpath(os.getcwd() + "/bd/categoriasImg"))
		self.dataLoader.cargarClases()
		self.setNumClasses(self.dataLoader.getNumClasses())
		self.classes = self.dataLoader.getClasses()
		self.threshold = 5e-8
		self.inputDim = 64

		self.batchSize = 40
		self.dataLoader.setBatchSize(self.batchSize)
		self.epochs = 10

	#def setEncoderDim(self, dim):
	#	self.clasificador.setEncoderDim(dim)
	#	self.autoencoder.setEncoderDim(dim)

	def setInputDim(self, dim):
		self.inputDim = dim

	def setNumClasses(self, num):
		self.numClasses = num


	def entrenar(self):
		
		self.dataLoader.cargarData()
		if(self.modeloIniciado == False):
			self.setInputDim(self.dataLoader.getInputDim())
			self.initModel()
		#exit()
		loss = 1
		lossBuscador = 1
		contador = 0
		#optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)




		while lossBuscador > self.threshold:
			self.trainingSet, self.labelsSet = self.dataLoader.nextTrainingData(labels=True)
			self.testingSet, self.testLabelSet = self.dataLoader.nextTestingData(labels=True)


			#loss = self.buscador.fit(self.trainingSet, self.labelsSet,
			#	batch_size=self.batchSize,
			#	epochs=self.epochs,
			#	verbose=0,
			#	validation_data=(self.testingSet, self.testLabelSet))
			loss = self.buscador.train_on_batch(self.trainingSet, self.labelsSet)
			score = self.buscador.evaluate(self.testingSet, self.testLabelSet, verbose=0)
			lossBuscador = score[0]
			#print('Test loss:', score[0])
			#print('Test accuracy:', score[1])

			#
			#lossBuscador = self.buscador.train_on_batch(self.trainingSet, self.labelsSet)
			#print("% Completado " + str(score[0]) + "            ", end='\r')
			print("% Completado " + str((self.threshold/lossBuscador) * 100) + "      loss: " + str(score[0]) + " accurracy " + str(score[1]), end='\r')
			contador += 1
			if contador%100 == 0:
				self.guardarAvance()
			if contador > 1000:
				contador = 0

	def guardarAvance(self):
		self.buscador.save_weights(os.path.normpath(os.getcwd() + "/models/pesos/model.h5"), True)	

	def initModel(self):
		if(self.modeloIniciado == True):
			return

		self.buscador = Sequential()
		self.buscador.add(Conv2D(64, kernel_size=(5, 5),
							activation='relu',
							data_format='channels_first',
							border_mode='same',
							input_shape=(1, self.inputDim, self.inputDim)))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		self.buscador.add(Conv2D(32, (10, 10), activation='relu'))

		

		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		self.buscador.add(Conv2D(16, (5, 5), activation='relu'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		self.buscador.add(Conv2D(8, (5, 5), activation='relu'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		self.buscador.add(Conv2D(4, (5, 5), activation='relu'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		self.buscador.add(Conv2D(2, (5, 5), activation='relu'))

		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		#self.buscador.add(Conv2D(12, (5, 5), activation='relu'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		#self.buscador.add(Conv2D(20, (5, 5), activation='relu'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		#self.buscador.add(Conv2D(12, (5, 5), activation='relu'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		#self.buscador.add(Conv2D(12, (1, 1), activation='relu'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		#self.buscador.add(Conv2D(64, (5, 5), activation='tanh'))
		#self.buscador.add(MaxPooling2D(pool_size=(2, 2)))
		#self.buscador.add(AveragePooling2D(pool_size=(2, 2)))
		#self.buscador.add(Dropout(0.25))
		self.buscador.add(Flatten())
		#self.buscador.add(Dense(128, activation='tanh'))
		
			#, kernel_regularizer=keras.regularizers.l2(0.01)
			#, activity_regularizer=keras.regularizers.l1(0.01)))
		#self.buscador.add(Dropout(0.25))
		#self.buscador.add(Dense(128, activation='tanh'))
			#, kernel_regularizer=keras.regularizers.l2(0.01)
			#, activity_regularizer=keras.regularizers.l1(0.01)))
		#self.buscador.add(Dropout(0.25))
		#self.buscador.add(Dense(128, activation='tanh'))
			#, kernel_regularizer=keras.regularizers.l2(0.01)
			#, activity_regularizer=keras.regularizers.l1(0.01)))
		#self.buscador.add(Dropout(0.25))
		#self.buscador.add(Dense(128, activation='tanh'))
			#, kernel_regularizer=keras.regularizers.l2(0.01)
			#, activity_regularizer=keras.regularizers.l1(0.01)))
		#self.buscador.add(Dropout(0.5))
		self.buscador.add(Dropout(0.25))
		
		self.buscador.add(Dense(self.numClasses, activation='softmax'))
		#self.buscador.summary()
		#exit()
		#optimizer = Adam(lr=5e-9, beta_1=0.5, beta_2=0.999)

		#este funciona!
		optimizer = keras.optimizers.Adadelta()
		self.buscador.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer
              ,metrics=['accuracy'])
		try:
			self.buscador.load_weights(os.path.normpath(os.getcwd() + "/models/pesos/model.h5"), True)
			print("pesos cargados")
		except OSError:
			print("no se han creado los pesos")
		self.modeloIniciado = True


	def search(self, pathImage):

		command_line = "python " + self.pathLib + " extract -v -i " + pathImage + " -o " + self.pathImgGenerada 

		p = subprocess.Popen(command_line, 
			shell=True, 
			stdout=subprocess.PIPE, 
			stderr=subprocess.PIPE) # Success!
		result = []
		for line in p.stdout:
			result.append(line)
			errcode = p.returncode
			print(errcode)
		for line in result:
			print(line)

		p.kill()

		filesList = []
		for subdir, dirs, files in os.walk(self.pathImgGenerada):
			for file in files:
				filesList.append(os.path.join(subdir, file))

		imgs = []
		for file in filesList:
			im = cv2.imread(file)
			im = cv2.resize(im,(64,64), interpolation = cv2.INTER_AREA)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			#cv2.imwrite(file,im)
			im = np.reshape(im, (1, 1, im.shape[0], im.shape[1]))

			if(self.modeloIniciado == False):
				self.setInputDim(im.shape[2])
				self.initModel()
			#imgs.append(im)

		#predicted = self.buscador.predict(imgs)
			print("prediccion " + str(self.classes[self.buscador.predict(im).argmax()]))