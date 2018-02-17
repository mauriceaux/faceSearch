class Escritor:
	def escribir(self, quien, mensaje):
		#print("[{0}] [#{1:05d}] loss_A: {2:.5f}, loss_B: {3:.5f}".format(time.strftime("%H:%M:%S"), iter, loss_A, loss_B),end='\r')
		print("mensaje de [{0}] : {1}".format(str(quien), str(mensaje)), end='\r')