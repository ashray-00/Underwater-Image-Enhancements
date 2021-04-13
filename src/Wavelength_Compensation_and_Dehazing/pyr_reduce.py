import numpy as np
import cv2
def pyr_reduce(img):
    kernelWidth = 5
    cw = 0.6
    ker1d = np.array([[0.25-cw/2,0.25,cw,0.25,0.25-cw/2]])
    kernel = np.kron(ker1d, ker1d)
    sz = img.shape
    imgout = []
    try:
        a = img.shape[2]
    except:
        a = 1
    for p in range(a):
        try:
            img1 = img[:,:,p]
        except:
            img1 = img
        imgFiltered = cv2.filter2D(img1, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        imgout.append(imgFiltered[0:sz[0]:2, 0:sz[1]:2])
    
    return np.array(imgout[0])