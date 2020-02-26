import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import norm
from scipy.signal import convolve2d as conv2
from scipy.ndimage import maximum_filter
import PIL.Image
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
def estimate_T(Jgdx, Jgdy, row, col, w, h):

    dxx = Jgdx * Jgdx #ok
    dxy = Jgdx * Jgdy
    dyy = Jgdy * Jgdy

    dxx = dxx[row - h//2:row + h//2, col - w//2:col + w//2]
    dxy = dxy[row - h//2:row + h//2, col - w//2:col + w//2]
    dyy = dyy[row - h//2:row + h//2, col - w//2:col + w//2]

    # INTEGRATE OVER AREAAAAAAAAA
    ii_dxx = np.sum(dxx)
    ii_dxy = np.sum(dxy)
    ii_dyy = np.sum(dyy)
    T = np.matrix([[ii_dxx, ii_dxy],
                   [ii_dxy, ii_dyy]])
    return T

def estimate_e(Ig, Jg, Jgdx, Jgdy, row, col, w, h):
    Ig = Ig[row-h//2:row+h//2, col-w//2:col+w//2]
    Jg = Jg[row-h//2:row+h//2, col-w//2:col+w//2]
    Jgdx = Jgdx[row-h//2:row+h//2, col-w//2:col+w//2]
    Jgdy = Jgdy[row-h//2:row+h//2, col-w//2:col+w//2]

    fx = (Ig - Jg) * Jgdx
    fy = (Ig - Jg) * Jgdy
    ii_fx = np.sum(fx)
    ii_fy = np.sum(fy)
    e = np.array([[ii_fx], [ii_fy]])

    return e

def shift_img(J, d):
    x = np.arange(d[0], J.shape[0] + d[0], 1)
    y = np.arange(d[1], J.shape[1] + d[1], 1)
    Jc = RectBivariateSpline(np.arange(J.shape[0]), np.arange(J.shape[1]), J)
    J_shifted = Jc(y,x)

    return J_shifted

#Return tensor field T = lambda*n*n^T
def orientation_tensor(img, gradksize, gradsigma, ksize, sigma):
    dont_use, Ig, Igdx, Igdy = regularized_values(img, img, gradksize, gradsigma)
    Tfield = np.zeros((img.shape[0], img.shape[1], 3))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            grad_I = np.array([[Igdx[x, y]], [Igdy[x,y]]])
            tensor = np.outer(grad_I, grad_I)
            Tfield[x,y,0] = tensor[0,0]
            Tfield[x,y,1] = tensor[1,0]
            Tfield[x,y,2] = tensor[1,1]

    lp, Tfield[:,:,0] = lowpass(Tfield[:,:,0], ksize, sigma)
    lp, Tfield[:,:,1] = lowpass(Tfield[:,:,1], ksize, sigma)
    lp, Tfield[:,:,2] = lowpass(Tfield[:,:,2], ksize, sigma)

    return Tfield

#Harris response
def harris(Tfield, kappa):

    Hfield = np.zeros((Tfield.shape[0], Tfield.shape[1]))

    for x in range(Tfield.shape[0]):
        for y in range(Tfield.shape[1]):
            tensor = np.matrix([[Tfield[x,y, 0], Tfield[x,y, 1]], [Tfield[x,y, 1], Tfield[x,y, 2]]])
            Hfield[x,y] = np.linalg.det(tensor) - kappa*np.trace(tensor)**2

    #print(Hfield)

    return Hfield

def get_n_harris_features(N, Hfield, threshold):
    #Set threshold
    img_thresh = Hfield > threshold

    #Select regions of interest
    img_select = img_thresh * Hfield

    #Max filter image
    img_max = maximum_filter(img_select, size=5)
    [row, col] = np.nonzero(img_select == img_max)

    #Create img of zeros with dots w values where corners are
    image = np.zeros((Hfield.shape[0], Hfield.shape[1]))
    image[row, col] = img_max[row, col]

    #Find N best features
    Hfield_values = image.flatten()
    values = np.sort(Hfield_values)
    values = values[::-1]
    indices = np.zeros((5,2), dtype =int)
    i = 0
    found_N = 0

    #print(values)

    while(found_N < N):
        N_max = values[i]
        #print(N_max)
        index = np.argwhere(abs(image - N_max) < 0.001)
        #print(index)

        i += 1

        if(index[0,0] > Hfield.shape[0]*0.9):
            continue
        elif(index[0,0] < Hfield.shape[0]*0.1):
            continue
        elif(index[0,1] > Hfield.shape[1]*0.9):
            continue
        elif(index[0,1] < Hfield.shape[1]*0.1):
            continue
        else:
            indices[found_N] = index
            found_N += 1
            

    #print(indices)
    return indices

def check_index_boundaries(x,y) :
    return True

#and (coord[0] > border) and
#(coord[1] < (Hfield.shape[1] - border)) and (coord[1] > border))
