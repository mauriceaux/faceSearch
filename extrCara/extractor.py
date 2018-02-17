from escritorConsola.escritorConsola import Escritor
import os
import shlex, subprocess
import shutil

class Extractor:
	

	def __init__(self, pathImagen, nomImagen=None):
		self.escritor = Escritor()
		
		self.pathImg = os.path.normpath(os.getcwd() + pathImagen)
		self.nomImagen = nomImagen
		self.pathImgGenerada = os.path.normpath(os.getcwd() + "/extrCara/gen")
		self.pathLib = os.path.normpath(os.getcwd() + "/lib/deepfakes/faceswap.py")
		self.escritor.escribir(str(self) + "_init_", "path es : " + str(self.pathImg))
		

	def setPathImgGenerada(self, path):
		self.pathImgGenerada = path

	def guardarImagen(self):
		#print(self.pathLib)
		#print(self.pathImg)
		#print(self.pathImgGenerada)
		#print(self.pathImgGenerada)

		#exit()

		command_line = "python " + self.pathLib + " extract -v -i " + self.pathImg + " -o " + self.pathImgGenerada 
		#command_line = "python " + self.pathLib
		#args = shlex.split(command_line)
		print(command_line)
		#exit()
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

		import cv2
		import numpy as np

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


		#hog = cv2.HOGDescriptor()




		if(None == self.nomImagen):
			filesList = []
			#print("creando lista de archivos")
			for subdir, dirs, files in os.walk(self.pathImgGenerada):
			    for file in files:
			    	filesList.append(os.path.join(subdir, file))
			    	#print("archivo agregado",os.path.join(subdir, file))
			#print("fin creando lista de archivos")
			#exit()
			for file in filesList:
				#print("intentando abrir " +file)
				im = cv2.imread(file)
				im = cv2.resize(im,(64,64), interpolation = cv2.INTER_AREA)
				im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				h = hog.compute(im)
				np.save(file + ".bin", h)
				#print("guardado " + file + ".bin")
				os.remove(file)
			return self.pathImgGenerada
		#exit()

		im = cv2.imread(self.pathImgGenerada + "\\" + self.nomImagen + "0.jpg")


		im = cv2.resize(im,(64,64), interpolation = cv2.INTER_AREA)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		h = hog.compute(im)
		np.save(self.pathImgGenerada + "\\" + self.nomImagen + ".bin", h)

		os.remove(self.pathImgGenerada + "\\" + self.nomImagen + "0.jpg")
		return self.pathImgGenerada + "\\" + self.nomImagen + ".bin.npy"

	def mostrarImagenOriginal(self):
		self.escritor.escribir(str(self) + "mostrarImagenOriginal", "mostrarImagenOriginal no implementado")


	def mostrarImagenGenerada(self):
		self.escritor.escribir(str(self) + "mostrarImagenGenerada", "mostrarImagenGenerada no implementado")		
