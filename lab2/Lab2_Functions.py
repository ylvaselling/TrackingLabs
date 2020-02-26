import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import norm
from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve as fftconvolve
from scipy.ndimage import maximum_filter
import PIL.Image
import sys
def load_lab_image(filename):
    """Load a grayscale image by filename from the CVL image directory

    Example:
    >>> img = load_lab_image('cornertest.png')
    """
    path = str(filename)
    return np.asarray(PIL.Image.open(path).convert('L'))

def get_cameraman():
    "Return I, J and true (col, row) displacement"
    n = 10 # Border crop
    img = plt.imread("images/cameraman.tif")
    I = img[n:-n, n:-n]
    x, y = 1, -2
    J = img[n-y:-n-y, n-x:-n-x]
    assert I.shape == J.shape
    return I, J, [[x], [y]]

def lowpass(img, ksize, sigma):
    lp = np.atleast_2d(np.exp(-0.5 * np.square(np.arange(-ksize/2,ksize/2,1)/sigma)))
    lp = lp / np.sum(lp)
    lp_img = conv2(conv2(img, lp, mode='same'), lp.T, mode='same')

    return lp, lp_img

def regularized_values(I, J, ksize, sigma):
    lp, Ig = lowpass(I, ksize, sigma)
    lp, Jg = lowpass(J, ksize, sigma)
    df = np.atleast_2d(-1.0 / np.square(sigma) * np.arange(-ksize/2,ksize/2,1) * lp)
    Jdgx = conv2(conv2(Jg, df, mode='same'), lp.T, mode='same')
    Jdgy = conv2(conv2(Jg, lp, mode='same'), df.T, mode='same')
    return Ig, Jg, Jdgx, Jdgy

# Structure tensor T at position (x,y) with window_size = [h,w]
def estimate_Tfield(Jgdx, Jgdy, w, h, ksize = 11, sigma = 2):

    dxx = Jgdx * Jgdx
    dxy = Jgdx * Jgdy
    dyy = Jgdy * Jgdy

    f_sum = np.ones((w,h))
    #lp = np.atleast_2d(np.exp(-0.5 * np.square(np.arange(-w/2,w/2,1)/sigma)))
    #f_sum = conv2(lp,lp.T)
   # print(lp.shape)
    conv_dxx = fftconvolve(dxx, f_sum, mode ='same')
    conv_dxy = fftconvolve(dxy, f_sum, mode ='same')
    conv_dyy = fftconvolve(dyy, f_sum, mode ='same')

    T_field = np.zeros((Jgdx.shape[0], Jgdx.shape[1], 2, 2))
    T_field[:,:,0, 0] = conv_dxx
    T_field[:,:,0, 1] = conv_dxy
    T_field[:,:,1, 0] = conv_dxy
    T_field[:,:,1, 1] = conv_dyy 
    

    return T_field

def estimate_efield(Ig, Jg, Jgdx, Jgdy, w, h, ksize = 11, sigma = 2):

    fx = (Ig - Jg) * Jgdx
    fy = (Ig - Jg) * Jgdy

    f_sum = np.ones((w,h))
    #lp = np.atleast_2d(np.exp(-0.5 * np.square(np.arange(-w/2,w/2,1)/sigma)))
    #f_sum = conv2(lp,lp.T)

    conv_fx = fftconvolve(fx, f_sum, mode ='same')
    conv_fy = fftconvolve(fy, f_sum, mode ='same')

    e_field = np.zeros((Jgdx.shape[0], Jgdx.shape[1], 2))
    e_field[:,:,0] = conv_fx
    e_field[:,:,1] = conv_fy

    return e_field


def shift_image_matrix(Vn, I, J):
    row_vec = np.arange(0,I.shape[0],1)
    col_vec = np.arange(0,I.shape[1],1)
    col, row = np.meshgrid(col_vec, row_vec)
    Jc = RectBivariateSpline(np.arange(J.shape[0]), np.arange(J.shape[1]), J)

    row = row+Vn[:,:,1]
    col = col+Vn[:,:,0]
    Jn = Jc.ev(row, col)

    return Jn