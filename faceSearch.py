from clasificador.clasificador import Clasificador
from encoder.encoder import Encoder
from data.loader import DataLoader
import keras
from keras.models import Sequential
from keras.optimizers import Adam


class faceSearch:

	def entrenar(self):
		self.clasificador = Clasificador()
		self.autoencoder = Encoder()
		self.dataLoader = DataLoader()
		self.dataLoader.setPathClassData(self.pathTrainingData)
		initModel()
		loss = 1
		while loss > self.threshold:
			self.trainingSet, self.labelsSet = self.dataLoader.nextTrainingData(labels=True)
			self.autoencoder.getEncoder().trainable(True)
			self.entrenador.train_on_batch(self.trainingSet, self.labelsSet)
			self.autoencoder.getEncoder().trainable(False)
			loss = self.validador.train_on_batch(self.trainingSet, self.trainingSet)
			print("% Completado " + str((self.umbral//loss) * 100), end="\r")



	def initModel(self):

		self.entrenador = Sequential()
		self.entrenador.add(self.autoencoder.getEncoder())
		self.entrenador.add(self.clasificador.getModel())

		self.validador = Sequential()
		self.validador.add(self.autoencoder.getEncoder())
		self.validador.add(self.autoencoder.getDecoder())

		optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

		self.entrenador.compile(loss='mean_absolute_error', optimizer=optimizer)
		self.validador.compile(loss='mean_absolute_error', optimizer=optimizer)
		self.autoencoder.getEncoder().compile(loss='mean_absolute_error', optimizer=optimizer)





	def search(self, pathImage):
		resultado = "soy el resultado"
		#IMPLEMENTAME
		return resultado