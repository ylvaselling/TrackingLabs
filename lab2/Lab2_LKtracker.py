import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import Lab2_Functions as f

def LK_equation(I, J, ksize, sigma, width, height):
    
    #Calculate blurred images and derivatives
    Ig, Jg, Jgdx, Jgdy = f.regularized_values(I, J, ksize, sigma)
 

    #Calculate T & E
    T_field = f.estimate_Tfield(Jgdx, Jgdy, width, height)
    e_field = f.estimate_efield(Ig, Jg, Jgdx, Jgdy, width, height)
   
    d_field = np.zeros((Ig.shape[0], Ig.shape[1], 2))
    #Td = e, returns d

    d_field = np.linalg.solve(T_field, e_field)
           

    return d_field
