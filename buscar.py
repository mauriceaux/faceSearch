import os
from extrCara.extractor import Extractor
from clasificador.clasificador import Clasificador
from encoder.encoder import Encoder
from escritorConsola.escritorConsola import Escritor
import numpy as np


path = "/fotoBuscar"
nomImagen = "seba"

extractor = Extractor(path, nomImagen)
extractedFacePath = extractor.guardarImagen()

#extractor.mostrarImagenOriginal()
#extractor.mostrarImagenGenerada()

encoder = Encoder(extractedFacePath)
encoder.setGuardarEstado(False)
repCara = encoder.predecir()
#exit()
#seba = os.path.normpath(os.getcwd() + "/clasificador/bd/categoriasEnc/seba/0.npy")
#papioncito = os.path.normpath(os.getcwd() + "/clasificador/bd/categoriasEnc/papioncito/0.npy")
#susti = os.path.normpath(os.getcwd() + "/clasificador/bd/categoriasEnc/susti/0.npy")
#willy = os.path.normpath(os.getcwd() + "/clasificador/bd/categoriasEnc/willi/2.npy")

clasificador = Clasificador()
clasificador.setEncoderDim(202)
clasificador.setBatchSize(4)
#respuesta = clasificador.buscar(seba)
respuesta = clasificador.buscar(repCara)
print(respuesta)
#respuesta = clasificador.buscar(seba)
#print(respuesta)
#respuesta = clasificador.buscar(willy)
#print(respuesta)
#respuesta = clasificador.buscar(susti)
#print(respuesta)
#escritor = Escritor()
#escritor.escribir("faceSearch", "la cara es de: " + str(respuesta))
