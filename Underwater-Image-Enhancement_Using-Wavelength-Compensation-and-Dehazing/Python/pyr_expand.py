import numpy as np
from scipy import signal
import cv2

def pyr_expand(i):
    kw = 5
    cw = 0.6
    ker1d = np.array([[0.25-cw/2,0.25,cw,0.25,0.25-cw/2]])
    kernel = np.kron(ker1d, np.transpose(ker1d)) * 4

    ker00 = kernel[0:kw:2, 0:kw:2]
    ker01 = kernel[0:kw:2, 1:kw:2]
    ker10 = kernel[1:kw:2, 0:kw:2]
    ker11 = kernel[1:kw:2, 1:kw:2]

    a = i.shape
    if (len(a) == 2):
        img = i[..., np.newaxis]
    else:
        img = i
    sz = img[:,:,0].shape
    osz = [0, 0]
    osz[0] = sz[0] * 2 - 1
    osz[1] = sz[1] * 2 - 1
    imgout = np.zeros(osz[0] * osz[1] * img.shape[2])
    imgout = imgout.reshape((osz[0], osz[1], img.shape[2]))
    # imgout = np.array([[[0 for i in range(img.shape[2])]for j in range(osz[1])] for k in range(osz[0])])

    for p in range(0, img.shape[2]):
        img1 = img[:,:,p]
        img1ph = np.pad(img1, ((0,0),(1,1)),'edge')
        img1pv = np.pad(img1, ((1,1),(0,0)),'edge')
        img00 = cv2.filter2D(img1, -1, ker00, borderType=cv2.BORDER_REPLICATE)
        img01 = signal.convolve2d(img1pv, ker01, mode='valid')
        img10 = signal.convolve2d(img1ph, ker10, mode='valid')
        img11 = signal.convolve2d(img1, ker11, mode='valid')
        imgout[0:osz[0]:2, 0:osz[1]:2,p] = img00
        imgout[1:osz[0]:2, 0:osz[1]:2,p] = img10
        imgout[0:osz[0]:2, 1:osz[1]:2,p] = img01
        imgout[1:osz[0]:2, 1:osz[1]:2,p] = img11

    return imgout