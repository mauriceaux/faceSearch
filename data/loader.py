from random import shuffle
import os
import shlex, subprocess
import shutil
import cv2
import numpy as np

class DataLoader:
	def __init__(self):
		self.classes = []
		self.numClasses = 0
		self.contador = 0
		self.contadorTest = 0
		self.batchSize = 30
		self.batchSizeTest = 10
		self.validationBatchSize = 40
		self.pathLib = os.path.normpath(os.getcwd() + "/lib/deepfakes/faceswap.py")
		self.pathParams = os.path.normpath(os.getcwd() + "/bd/param/")
		self.validationSize = 0.1
		try:
			self.inputDim = np.asscalar(np.load(self.pathParams + "\\inputDim.npy"))
		except FileNotFoundError:
			self.inputDim = 0
		try:
			self.dataPaths = np.load(self.pathParams + "\\dataPaths.npy").tolist()
		except FileNotFoundError:
			self.dataPaths = []
		try:
			self.dataPathsTest = np.load(self.pathParams + "\\dataPathsTest.npy").tolist()
		except FileNotFoundError:
			self.dataPathsTest = []
		

	def getInputDim(self):
		print("input dim desde loader ", self.inputDim)
		return self.inputDim


	def setPathTrainingData(self, path):
		self.pathTrainingData = path

	def getClasses(self):
		print("clases encontradas ", self.classes)
		return self.classes

	def getNumClasses(self):
		return self.numClasses

	def setPathClassData(self, path):
		self.pathClassData = path

	def cargarClases(self):
		self.classes = self.extraerNombreSubCarpetas()
		self.numClasses = len(self.classes)
		
	def extraerNombreSubCarpetas(self):
		return [dI for dI in os.listdir(self.pathTrainingData) if os.path.isdir(os.path.join(self.pathTrainingData,dI))]

	def setBatchSize(self, batchSize):
		self.batchSize = batchSize

	def actualizarListas(self):
		shuffle(self.dataPaths)
		shuffle(self.dataPathsTest)

	def nextTrainingData(self, labels=True):
		self.actualizarListas()
		trainingData = []
		trainingLabels = []
		for i in  range(0,self.batchSize):
			if(self.contador > len(self.dataPaths)-1):
				self.contador = 0
			
			#print("imagen training", self.dataPaths[self.contador])
			data = cv2.imread(self.dataPaths[self.contador] )
			data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
			data = np.reshape(data, (1, data.shape[0], data.shape[1]))
			trainingData.append(data)
			validClass = np.zeros((self.numClasses), np.uint8)
			validClass[self.classes.index(os.path.basename(os.path.dirname(self.dataPaths[self.contador])))] = 1
			#print("clase", self.classes[self.classes.index(os.path.basename(os.path.dirname(self.dataPaths[self.contador])))])
			trainingLabels.append(validClass)
			self.contador = self.contador + 1
		trainingData = np.asarray(trainingData)
		trainingLabels = np.asarray(trainingLabels)
		return trainingData, trainingLabels

	def nextTestingData(self, labels=True):
		self.actualizarListas()
		trainingData = []
		trainingLabels = []
		for i in  range(0,self.batchSizeTest):
			if(self.contadorTest > len(self.dataPathsTest)-1):
				self.contadorTest = 0
			
			#print("imagen testing", self.dataPathsTest[self.contadorTest])
			data = cv2.imread(self.dataPathsTest[self.contadorTest] )
			data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
			data = np.reshape(data, (1, data.shape[0], data.shape[1]))
			trainingData.append(data)
			validClass = np.zeros((self.numClasses), np.uint8)
			validClass[self.classes.index(os.path.basename(os.path.dirname(self.dataPathsTest[self.contadorTest])))] = 1
			#print("clase", self.classes[self.classes.index(os.path.basename(os.path.dirname(self.dataPathsTest[self.contador])))])
			trainingLabels.append(validClass)
			self.contadorTest = self.contadorTest + 1
		trainingData = np.asarray(trainingData)
		trainingLabels = np.asarray(trainingLabels)
		return trainingData, trainingLabels



	def cargarData(self):
		dataPaths = []
		for className in  self.classes:
			pathImage = os.path.normpath(os.getcwd() + "/bd/categoriasImg/" + className)
			pathImageFace = os.path.normpath(os.getcwd() + "/bd/caras/" + className)
			if not os.path.exists(pathImageFace):
				command_line = "python " + self.pathLib + " extract -v -i " + pathImage + " -o " + pathImageFace

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
			for subdir, dirs, files in os.walk(pathImageFace):
				for file in files:
					filesList.append(os.path.join(subdir, file))

			

			for file in filesList:
				im = cv2.imread(file)
				im = cv2.resize(im,(64, 64), interpolation = cv2.INTER_AREA)
				im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				#im = np.reshape(im, (1, im.shape[0], im.shape[1]))
				cv2.imwrite(file,im)
				if self.inputDim == 0:
					self.inputDim = 64
				dataPaths.append(file)

		shuffle(dataPaths)
		#print("data paths", dataPaths)
		#dataPaths = shuffle(dataPaths)

		self.dataPaths = dataPaths[int(len(dataPaths) * .00) : int(len(dataPaths) * (1-self.validationSize))]
		self.dataPathsTest = dataPaths[int(1 + (len(dataPaths) * (1-self.validationSize))) : len(dataPaths)]
		np.save(self.pathParams + "\\dataPaths", np.asarray(self.dataPaths))
		np.save(self.pathParams + "\\dataPathsTest", np.asarray(self.dataPathsTest))
		np.save(self.pathParams + "\\inputDim", np.asarray(self.inputDim))