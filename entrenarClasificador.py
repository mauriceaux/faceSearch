from clasificador.clasificador import Clasificador

clasificador = Clasificador()
clasificador.generarEnc()
clasificador.setBatchSize(4)
clasificador.setEncoderDim(202)
clasificador.entrenar()