from models.clasificador  import Clasificador
from models.autoencoder import Autoencoder
from data.loader import DataLoader
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from data.loader import DataLoader

import os
import shlex, subprocess
import shutil
import cv2
import numpy as np


class faceSearch:

	def __init__(self):
		print("init")
		self.clasificador = Clasificador()
		self.autoencoder = Autoencoder()
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
		self.optimizer = None

	def setEncoderDim(self, dim):
		self.clasificador.setEncoderDim(dim)
		self.autoencoder.setEncoderDim(dim)

	def setInputDim(self, dim):
		self.autoencoder.setInputDim(dim)

	def setNumClasses(self, num):
		self.clasificador.setNumClasses(num)


	def entrenar(self):
		
		self.dataLoader.generarHogData()
		if(self.modeloIniciado == False):
			self.setInputDim(self.dataLoader.getInputDim())
			#print("self.dataLoader.getInputDim()", self.dataLoader.getInputDim())
			#exit()
			self.setEncoderDim(int(self.dataLoader.getInputDim()/10))
			self.initModel()
		#exit()
		loss = 1
		lossBuscador = [1,0]
		contador = 0
		optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
		while lossBuscador[0] > self.threshold:
			self.trainingSet, self.labelsSet = self.dataLoader.nextTrainingData(labels=True)
			#self.autoencoder.encoder.trainable = True
			self.autoencoder.encoder.compile(loss='mean_absolute_error', optimizer=optimizer)
			lossBuscador = self.buscador.train_on_batch(self.trainingSet, self.labelsSet)
			#self.autoencoder.encoder.trainable = False
			#self.autoencoder.encoder.compile(loss='mean_absolute_error', optimizer=optimizer)
			#loss = self.validador.train_on_batch(self.trainingSet, self.trainingSet)
			print("% Completado " + str((self.threshold/lossBuscador[0]) * 100), end='\r')
			#print("% Completado " + str((self.threshold//loss) * 100) + " loss buscador: " + str(lossBuscador), end='\r')
			contador += 1
			if contador%100 == 0:
				self.guardarAvance()
			if contador > 1000:
				contador = 0

	def guardarAvance(self):
		self.buscador.save_weights(os.path.normpath(os.getcwd() + "/models/pesos/buscador.h5"), True)	
		self.validador.save_weights(os.path.normpath(os.getcwd() + "/models/pesos/validador.h5"), True)	


	def initModel(self):
		if(self.modeloIniciado == True):
			return

		optimizerEncoder = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
		self.autoencoder.getEncoder().compile(loss='mean_absolute_error', optimizer=optimizerEncoder)

		optimizerClasificador = Adam(lr=5e-5, beta_1=0.1, beta_2=0.999)
		self.clasificador.getModel().compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])
		

		self.buscador = Sequential()
		self.buscador.add(self.autoencoder.getEncoder())
		self.buscador.add(self.clasificador.getModel())

		self.validador = Sequential()
		self.validador.add(self.autoencoder.getEncoder())
		self.validador.add(self.autoencoder.getDecoder())

		optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

		self.buscador.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])
		#self.validador.compile(loss='mean_absolute_error', optimizer=optimizer)
		try:
			self.buscador.load_weights(os.path.normpath(os.getcwd() + "/models/pesos/buscador.h5"), True)
			self.validador.load_weights(os.path.normpath(os.getcwd() + "/models/pesos/validador.h5"), True)
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

		winSize = (20,20)
		blockSize = (10,10)
		blockStride = (5,5)
		cellSize = (10,10)
		nbins = 9
		derivAperture = 1
		winSigma = -1.
		histogramNormType = 0
		L2HysThreshold = 0.2
		gammaCorrection = 1
		nlevels = 64
		signedGradients = True

		hog = cv2.HOGDescriptor(winSize
			,blockSize
			,blockStride
			,cellSize
			,nbins
			,derivAperture
			,winSigma
			,histogramNormType
			,L2HysThreshold
			,gammaCorrection
			,nlevels
			,signedGradients)

		filesList = []
		for subdir, dirs, files in os.walk(self.pathImgGenerada):
			for file in files:
				filesList.append(os.path.join(subdir, file))

		for file in filesList:
			im = cv2.imread(file)
			im = cv2.resize(im,(64,64), interpolation = cv2.INTER_AREA)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			h = hog.compute(im)

			if(self.modeloIniciado == False):
				self.setInputDim(h.shape[0])
				self.setEncoderDim(h.shape[0]//10)
				self.initModel()
			h = np.reshape(h, (h.shape[1], h.shape[0]))
			predicted = self.buscador.predict(h)
			#print("prediccion", str(predicted))
			#for i in range(0, len(self.classes)):
			#	print(self.classes[i] + " -> " + str(predicted[0][i]))
			#print(str(self.classes[0]))
			print("prediccion " + str(self.classes[self.buscador.predict(h).argmax()]))




		resultado = "soy el resultado"
		#IMPLEMENTAME
		return resultado