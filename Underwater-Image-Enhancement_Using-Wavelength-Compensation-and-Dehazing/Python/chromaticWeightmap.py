import numpy as np
import cv2

def chromaticWeightmap(im):
    hsvImage = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    saturationValue = hsvImage[:,:,1]
    saturationMax = 1
    sigma = 0.3
    output = np.exp(-1 * np.square(saturationValue - saturationMax) / (2 * sigma**2))
    return output