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

	def __init__(self):
		self.escritor = Escritor()
		self.cargar()
		self.escritor.escribir(str(self) + "_init_", "clasificador creado")
		self.pathBD = "/clasificador/bd"
		self.absPathBD = os.path.normpath(os.getcwd() + "/clasificador/bd")
		self.encoderDim = 0
		self.batchSize = 0
		self.validationSize = 0.2
		self.trainingSet = []
		self.testingSet = []
		self.posActualTrainingData = 0
		self.posActualTestingData = 0
		self.classes = []
		self.numClasses = 0
		subdirList = [dI for dI in os.listdir(self.absPathBD + "/categoriasEnc") if os.path.isdir(os.path.join(self.absPathBD+ "/categoriasEnc",dI))]
		for subdir in subdirList:
			self.classes.append(subdir)
		self.numClasses = len(self.classes)
		#self.umbral = 5e-5
		self.umbral = 0.00002
		self.limpiarCarpetas()
		self.model = None


	def limpiarCarpetas(self):
		subdirList = [dI for dI in os.listdir(self.absPathBD + "/categoriasHog") if os.path.isdir(os.path.join(self.absPathBD+ "/categoriasHog",dI))]
		for subdir in subdirList:
			shutil.rmtree(os.path.normpath(self.absPathBD + "/categoriasHog/" + subdir))


	def setBatchSize(self, batchSize):
		self.batchSize = batchSize

	def setEncoderDim(self, dim):
		self.encoderDim = dim


	def generarEnc(self):

		hogEncodedList = []
		subdirList = [dI for dI in os.listdir(self.absPathBD + "/categoriasImg") if os.path.isdir(os.path.join(self.absPathBD+ "/categoriasImg",dI))]

		for folder in subdirList:
			extractor = Extractor(self.pathBD + "/categoriasImg/" + folder)
			extractor.setPathImgGenerada(os.path.normpath(os.getcwd() + self.pathBD + "/categoriasHog/" + folder))
			encodedFacePath = extractor.guardarImagen()
			hogEncodedList.append(encodedFacePath)

		encoder = Encoder(os.path.normpath(os.getcwd() + self.pathBD + "/categoriasEnc"))
		encoder.setBatchSize(10)

		for encoded in hogEncodedList:
			for subdir, dirs, files in os.walk(encoded):
				#encoder.setModelName(os.path.basename(encoded))
				encoder.setPath(subdir)
				encoder.setPathEncodedData(os.path.normpath(self.absPathBD + "/categoriasEnc/" + os.path.basename(encoded) + "/" ))
				encoder.initModel()
				encoder.generar()
				#print("repCara.shape " + str(repCara.shape))
				#exit()
				#for file in files:
					#encoder.setPath(subdir)
					#print(os.path.join(subdir, file))
					#encoder = Encoder(os.path.join(subdir, file))
					#print(os.path.basename(encoded))
					#print(encoded)
					#exit()
					#encoder.setPathEncodedData(os.path.normpath(os.getcwd() + self.pathBD + "/categoriasEnc/" + os.path.basename(encoded) + "/" + file))
					#repCara = encoder.generar()

	def entrenar(self):
		
		model = self.initModel()
		try:
			model.load_weights(os.path.normpath(os.getcwd() + "/clasificador/model/model.h5"))
			print("pesos cargados")
		except OSError:
			print("no se han creado los pesos")


		#X_train, y_train, X_test, y_test = self.load_data()
		#self.encoderDim = X_train[0].shape
		epochs = 500
		self.umbral = 5e-05
		loss = [1.0, 0]
		inicio = time.time()
		while loss[0] > self.umbral:
		#for i in range(0,epochs):
			X_train, y_train, X_test, y_test = self.load_data()
			#print("X_train.shape" + str(X_train.shape))
			#print("X_test.shape" + str(X_test.shape))
			#print("y_train.shape" + str(y_train.shape))
			#print("y_test.shape" + str(y_test.shape))
			self.encoderDim = X_train[0].shape
			loss = model.train_on_batch(X_train, X_test)
			score = model.evaluate(y_train, y_test, verbose=0)
			porcentajeCompletado = (100 * self.umbral)/loss[0]
			self.escritor.escribir("entrenamiento clasificador loss ", porcentajeCompletado)
			#print("                                                                             ",end='\r')
			#print("score " + str(score) + " \tloss " + str(loss),end="\r")
			#print("loss",model.train_on_batch(np.asarray(X_train), np.asarray(X_test)))
		fin = time.time()
		print("\n")
		print("fin entrenamiento, demoro " + str(fin-inicio) + " segundos")
		model.save_weights(os.path.normpath(os.getcwd() + "/clasificador/model/model.h5"), True)

	def buscar(self, imageRep):
		model = self.initModel()		
		try:
			model.load_weights(os.path.normpath(os.getcwd() + "/clasificador/model/model.h5"))
			print("pesos cargados")
		except OSError:
			print("no se han creado los pesos")

		
		rep = np.load(imageRep)
		imagen = np.zeros((rep.shape[0], 1), np.float32)
		imagen = rep[:]
		#rep  = rep[:,]
		imagen = np.reshape(rep, (rep.size, 1))
		#print("rep.shape " + str(imagen.shape))
		imagen = imagen.reshape(imagen.shape[1], imagen.shape[0])
		#print("rep.shape" + str(imagen.shape))

		#exit()
		#print("imageRep.shape",rep.shape)
		#exit()

		#fixed = rep.reshape(rep.shape[1], rep.shape[0])
		#final = fixed[0]
		prediccion = model.predict(imagen)
		#print(prediccion)
		#print(self.classes[prediccion.argmax()])
		return self.classes[prediccion.argmax()]





	def load_data(self):
		self.actualizarListas()
		X_train = []
		y_train = []
		X_test = []
		y_test = []
		for i in  range(0,self.batchSize):
			if(self.posActualTrainingData > len(self.trainingSet)-1):
				self.posActualTrainingData = 0
			if(self.posActualTestingData > len(self.testingSet)-1):
				self.posActualTestingData = 0
			#print("self.trainingSet.shape", len(self.trainingSet))
			#print("self.posActualTrainingData", self.posActualTrainingData)
			#exit()
			X_train.append(np.load(self.trainingSet[self.posActualTrainingData]))
			x_valid = np.zeros((len(self.classes)), np.uint8)
			
			#print("self.posActualTestingData ", str(self.posActualTrainingData))
			x_valid[self.classes.index(os.path.basename(os.path.dirname(self.trainingSet[self.posActualTrainingData])))] = 1
			X_test.append(x_valid)
			#X_test.append(os.path.basename(os.path.basename(os.path.dirname(self.trainingSet[self.posActualTrainingData]))))
			y_train.append(np.load(self.testingSet[self.posActualTestingData]))
			y_valid = np.zeros((len(self.classes)), np.uint8)

			y_valid[self.classes.index(os.path.basename(os.path.dirname(self.testingSet[self.posActualTestingData])))] = 1
			#print("self.trainingSet[self.posActualTrainingData] " + str(self.testingSet[self.posActualTestingData]))
			#print("self.classes " + str(self.classes))
			#print("os.path.basename(os.path.basename(os.path.dirname(self.testingSet[self.posActualTrainingData]))) " + str(os.path.basename(os.path.basename(os.path.dirname(self.testingSet[self.posActualTestingData])))))
			#print(self.testingSet)
			#exit()
			y_test.append(y_valid)
			self.posActualTrainingData += 1
			self.posActualTestingData += 1

		X_train = np.asarray(X_train)
		#print("X_train.shape " + str(X_train.shape))
		#exit()
		#X_train = X_train.reshape((X_train.shape[0], X_train.shape[2]))
		
		y_train = np.asarray(y_train)
		#y_train = y_train.reshape((y_train.shape[0], y_train.shape[2]))

		X_test = np.asarray(X_test)
		#print("X_test.shape",X_test.shape)
		#exit()
		#X_test = X_test.reshape((X_test.shape[0]))
		y_test = np.asarray(y_test)
		#print("y_test.shape" + str(y_test))
		#exit()
		#y_test = y_test.reshape((y_test.shape[0]))
		return X_train, y_train, X_test, y_test


	def actualizarListas(self):
		lista = []
		for subdir, dirs, files in os.walk(os.path.normpath(os.getcwd() + self.pathBD + "/categoriasEnc/")):
			for file in files:
				lista.append(os.path.join(subdir, file))
				#validationSetSize = len(self.testingSet)/len(files)
				#if(validationSetSize == 0 or validationSetSize < self.validationSize):
				#	self.testingSet.append(os.path.join(subdir, file))
				#else:
				#	self.trainingSet.append(os.path.join(subdir, file))
		shuffle(lista)
		for i in range(0,len(lista)):
			validationSetSize = len(self.testingSet)/len(lista)
			if(validationSetSize == 0 or validationSetSize < self.validationSize):
				self.testingSet.append(lista[i])
			else:
				self.trainingSet.append(lista[i])

		
		shuffle(self.trainingSet)
		shuffle(self.testingSet)





	def initModel(self):
		self.encoderDim


		optimizer = Adam(lr=5e-5, beta_1=0.1, beta_2=0.999)
		self.model = Sequential()
		self.model.add(Dense(input_dim=(self.encoderDim), output_dim=(self.encoderDim * 2)))
		self.model.add(Activation('relu'))
		self.model.add(Dense(output_dim=(self.encoderDim * 2)))
		self.model.add(Activation('relu'))
		self.model.add(Dense(output_dim=(self.encoderDim * 2)))
		self.model.add(Activation('relu'))
		self.model.add(Dense(output_dim=self.numClasses))
		self.model.add(Activation('softmax'))
		#model.compile(loss='mean_squared_error', optimizer=optimizer)

		self.model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])
		return model


	def getModel(self)
		return self.model

	def cargar(self):
		self.escritor.escribir(str(self) + "cargar", "debes entrenarme primero!")

	#def buscar(self, rep):
	#	self.escritor.escribir(str(self) + "buscar", "buscando " + str(rep))
	#	return "resultado de la busqueda"

