import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

image = plt.imread("image.png")
plt.imshow(image)
plt.show()

h, w, c  = image.shape  # colors, height, width
print(image.shape)

regions = 10

hrange = np.arange(h)
wrange = np.arange(w)
MGx, MGy = np.meshgrid(wrange, hrange)

for r in range(regions):
    regionh = np.arange((r*h/regions),((r+1)*h/regions))
    regionw = np.arange((r*w/regions),((r+1)*w/regions))
    Rx, Ry = np.meshgrid(regionw, regionh)
    new_image = np.empty_like(image)

    for channel in range(c):
        # Get a linear interpolation for this color channel.
        interpolation = RectBivariateSpline(hrange, wrange, image[:,:,channel], kx=2, ky=2)
        derivatives = interpolation(regionh,regionw, 1, grid=True)
        print(derivatives.shape)
        height, width = derivatives.shape
        print("BREAK")
        sum = 0;
        for i in range(height):
            for j in range(width):
                sum += np.abs(derivatives[i,j])
        #print(sum)
        # grid = False since the deformed grid is irregular
        new_image[:,:,channel] = interpolation(MGy, MGx, grid=False)

plt.imshow(new_image)
plt.show()
