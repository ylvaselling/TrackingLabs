import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import Lab2_Functions as f
from Lab2_LKtracker import LK_equation
from scipy.interpolate import RectBivariateSpline
from scipy.signal import medfilt
import Lab2_GivenFunctions as lab2

def LK_equation_multiscale(I,J,ksize,sigma, w, h, number_of_scales):

	V_tot = np.zeros((I.shape[0], I.shape[1], 2))
	Jn = J

	for n in range(number_of_scales, 0 , -1):
		sc = 2**(n-1)
		Vn =  LK_equation(I,Jn,sc*ksize, sc*sigma, sc*w, sc*h)

		#Remove outliers
		#if(sc*ksize%2 == 0):
			#Vn[:,:,0] = medfilt(Vn[:,:,0], sc*ksize+1)
			#Vn[:,:,1] = medfilt(Vn[:,:,1], sc*ksize+1)
		#else:
			#Vn[:,:,0] = medfilt(Vn[:,:,0], sc*ksize)
			#Vn[:,:,1] = medfilt(Vn[:,:,1], sc*ksize)

		V_tot += Vn


		Jn = f.shift_image_matrix(V_tot, I ,J)


	lab2.gopimage(Vn)
	plt.show()
	lab2.gopimage(V_tot)
	plt.show()

	plt.imshow(Jn)
	plt.show()
	plt.imshow(np.abs(Jn-I))
	plt.show()

	return V_tot
