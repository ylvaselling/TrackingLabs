import numpy as np
from matplotlib import pyplot as plt
from Lab2_LKtracker import LK_equation
import Lab2_Functions as f
from Lab2_LK_equation_multiscale import LK_equation_multiscale
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import norm
import Lab2_GivenFunctions as lab2

#Load chessboard
img1 = f.load_lab_image("labfiles/forwardL0.png")
folder = "labfiles/forwardL"
noOfImg = 10
images = np.zeros((img1.shape[0], img1.shape[1], noOfImg))

#Load images
for i in range(noOfImg):
    imgnumber = i
    filename = folder + str(imgnumber) + ".png"
    img = f.load_lab_image(filename)
    images[:,:,i] = img #ok

#Part 1.1 - Single scale LK
I_cam, J_cam, dTrue = f.get_cameraman()
Vn_cam = LK_equation(I_cam, J_cam, 11, 2, 40, 40)
Jn_cam = f.shift_image_matrix(Vn_cam, I_cam, J_cam)

lab2.gopimage(Vn_cam)
plt.show()
#Part 1.2
I = images[:,:,0]
J = images[:,:,1]
Vn =  LK_equation( I, J,11, 2, 40, 40)

Jn = f.shift_image_matrix(Vn, I, J)
lab2.gopimage(Vn)
plt.show()

if(norm(J-I) > norm(Jn-I)):
	print("norm(Jn-I) is smaller than norm(J-I)")

#Part 2 - Multi scale LK
#Load car
'''
img1 = f.load_lab_image("labfiles/SCcar4_00070.bmp")
folder = "labfiles/SCcar4_0007"
noOfImg = 20
images = np.zeros((img1.shape[0], img1.shape[1], noOfImg))

#Load images
for i in range(noOfImg):
	if(i == 10):
		folder = "labfiles/SCcar4_0008"

	imgnumber = i%10
	filename = folder + str(imgnumber) + ".bmp"
	img = f.load_lab_image(filename)
	images[:,:,i] = img #ok
'''
V_tot = np.zeros((img1.shape[0], img1.shape[1],2))

for i in range(noOfImg-1):

	Vn = LK_equation_multiscale(images[:,:,i], images[:,:,i+1], 11, 2, 40, 40, 5)
	V_tot += Vn
