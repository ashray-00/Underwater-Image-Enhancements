import cv2
import numpy as np
from scipy.signal import convolve2d

def saliencyWeightmap(im):
    if (im.shape[2] > 1):
        imGray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    else:
        imGray = im
    
    kernel_1D = 1/16 * np.array([[1, 4, 6, 4, 1]])
    kernel_2D = np.kron(kernel_1D, np.transpose(kernel_1D))
    I_mean = np.mean(imGray[:])

    I_Whc = convolve2d(imGray, kernel_2D, mode='same')

    output = np.abs(I_Whc - I_mean)

    return output