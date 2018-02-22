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
		self.batchSize = 20
		self.pathLib = os.path.normpath(os.getcwd() + "/lib/deepfakes/faceswap.py")
		self.pathParams = os.path.normpath(os.getcwd() + "/bd/param/")
		try:
			self.inputDim = np.asscalar(np.load(self.pathParams + "\\inputDim.npy"))
		except FileNotFoundError:
			self.inputDim = 0
		try:
			self.dataPaths = np.load(self.pathParams + "\\dataPaths.npy")
		except FileNotFoundError:
			self.dataPaths = []
		

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

	def actualizarListas(self):
		shuffle(self.dataPaths)

	def nextTrainingData(self, labels=True):
		self.actualizarListas()
		trainingData = []
		trainingLabels = []
		for i in  range(0,self.batchSize):
			if(self.contador > len(self.dataPaths)-1):
				self.contador = 0
			
			data = np.load(self.dataPaths[self.contador] + ".npy")
			data = np.reshape(data, (data.shape[1], data.shape[0]))
			trainingData.append(data[0])
			validClass = np.zeros((self.numClasses), np.uint8)
			validClass[self.classes.index(os.path.basename(os.path.dirname(self.dataPaths[self.contador])))] = 1
			#print("datos de " + str(self.dataPaths[self.contador] + ".npy"))
			#print("valid clase " + str( self.classes[self.classes.index(os.path.basename(os.path.dirname(self.dataPaths[self.contador])))] ))
			trainingLabels.append(validClass)
			self.contador += 1
		trainingData = np.asarray(trainingData)
		trainingLabels = np.asarray(trainingLabels)
		return trainingData, trainingLabels
		#print(trainingData.shape)
		#exit()



	def generarHogData(self):
		for className in  self.classes:
			pathImage = os.path.normpath(os.getcwd() + "/bd/categoriasImg/" + className)
			pathImageFace = os.path.normpath(os.getcwd() + "/bd/caras/" + className)
			pathImageHog = os.path.normpath(os.getcwd() + "/bd/categoriasHog/" + className)
			if not os.path.exists(pathImageHog):
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
				for subdir, dirs, files in os.walk(pathImageFace):
					for file in files:
						filesList.append([os.path.join(subdir, file), file])

				for file in filesList:
					im = cv2.imread(file[0])
					im = cv2.resize(im,(64,64), interpolation = cv2.INTER_AREA)
					im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
					h = hog.compute(im)
					if self.inputDim == 0:
						self.inputDim = h.shape[0]
						np.save(self.pathParams + "\\inputDim", self.inputDim)
					if not os.path.exists(pathImageHog):
						os.makedirs(pathImageHog)
					np.save(pathImageHog + "\\" + file[1], h)
					self.dataPaths.append(pathImageHog + "\\" + file[1])
					#np.append(self.dataPaths, pathImageHog + "\\" + file[1])
					np.save(self.pathParams + "\\dataPaths", np.asarray(self.dataPaths))