class DataLoader:
	def setPathClassData(self, pathTrainingData):
		self.pathTrainingData = pathTrainingData
		self.cargarClases()

	def selfCargarClases(self):
		self.classes = self.extraerNombreSubCarpetas(self.pathTrainingData)
		self.numClasses = len(self.classes)
		