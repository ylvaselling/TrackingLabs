import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import Lab1_Functions as f

def LK_tracker(I, J, ksize, sigma, row, col, width, height):
    #Displacement vectors
    d_tot = np.zeros((2,1))
    d = np.ones((2,1))
    #Calculate blurred images and derivatives
    Ig, Jg, Jgdx, Jgdy = f.regularized_values(I, J, ksize, sigma)

    while norm(d) > 0.1 :
        #Calculate T & E
        T = f.estimate_T(Jgdx, Jgdy, row, col, width, height)
        e = f.estimate_e(Ig, Jg, Jgdx, Jgdy, row, col, width, height)
        #Td = e, returns d
        d = np.linalg.solve(T, e)
        #Shift images
        J = f.shift_img(J, d)
        Jgdx = f.shift_img(Jgdx, d)
        Jgdy = f.shift_img(Jgdy, d)
        Jg = f.shift_img(Jg, d)
        #Update d_tot
        d_tot += d

    return d_tot
