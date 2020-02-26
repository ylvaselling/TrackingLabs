import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
from Lab1_LKtracker import LK_tracker
from numpy.linalg import norm
import Lab1_Functions as f
import os

#Test LK-tracker
#I, J, dTrue = f.get_cameraman()

#(I, J, ksize, sigma, row, col, width, height)
#result = LK_tracker(I, J, 11, 2, 85, 120, 40, 70) #Index ok
#assert norm(result - dTrue) < 0.1,"Error with LK-tracker!"

img1 = f.load_lab_image("images/chessboard/img1.png")
filename = "images/chessboard/img"
noOfImg = 10
images = np.zeros((img1.shape[0], img1.shape[1], noOfImg))

#Load images
for i in range(noOfImg):
    imgnumber = i + 1
    filename = "images/chessboard/img" + str(imgnumber) + ".png"
    img = f.load_lab_image(filename)
    images[:,:,i] = img #ok

#Calculate tensor field
print("Calculating orientation tensor...")
Tfield = f.orientation_tensor(images[:,:,0], 9, 2, 9, 2)
#Calculate Harris field
print("Calculating harris field...")
Hfield = f.harris(Tfield, 0.05)

#Find 5 best feature points
#N, Hfield, threshold
indices = f.get_n_harris_features(5, Hfield, 10000)
#Empty array for saving displacement vectors
d_tot = np.zeros((indices.shape[0], indices.shape[1]))
d = np.zeros((indices.shape[0], indices.shape[1]))

#Tracking
for i in range(noOfImg-1):

    #Convert floating displacemenet values to integers
    d_index = np.rint(d, casting='unsafe').astype(int,casting='unsafe')

    #Add displacement to incices
    #print("d indices bf & after displacement")
    print(d_index)
    indices += d_index

    #Feature index
    j = 1
    #Plot image and feature points
    plt.imshow(images[:,:,i])
    plt.scatter(indices[:,1], indices[:,0], c='blue') #Index ok
    #plt.scatter(indices[j,1], indices[j,0], c='red')
    plt.show()

    #print(indices[j,0])
    #print(indices[j,1])

    for j in range(5):
        #Track index w LK tracker
        #x = col, y = row
        #I, J, ksize, sigma, row, col, width, height
        #Returns x, y displacement
        result = LK_tracker(images[:,:,i], images[:,:,i+1], 11, 2, indices[j,0] , indices[j,1] , 60, 60) #Index ok
        #print("result from LK tracker")
        #print(result)
        result = result.flatten()
        d[j, :] = result[::-1]
        #print("d")
        #print(d)

        #Add displacement to total displacement
        d_tot += d
        #print("d tot")
        #print(d_tot)

print(d_tot)
