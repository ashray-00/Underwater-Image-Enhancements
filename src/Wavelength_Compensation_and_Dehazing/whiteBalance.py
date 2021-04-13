import numpy as np

def whiteBalance(im):
    R_avg = np.mean(im[:,:,0])
    G_avg = np.mean(im[:,:,1])
    B_avg = np.mean(im[:,:,2])

    RGB_avg = [R_avg, G_avg, B_avg]

    gray_value = (R_avg + G_avg + B_avg) / 3

    scaleValue = gray_value / RGB_avg

    w = np.zeros_like(im)
    w[:,:,0] = scaleValue[0] * im[:,:,0]
    w[:,:,1] = scaleValue[1] * im[:,:,1]
    w[:,:,2] = scaleValue[2] * im[:,:,2]

    return w
