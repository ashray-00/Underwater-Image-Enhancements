import numpy as np

def enhanceContrast(im):
    luminance = im[:,:,0] * 0.299 + im[:,:,1] * 0.587 + im[:,:,2] * 0.114
    avgLuminance = np.mean(luminance[:])
    gamma = 2 * (0.5 + avgLuminance)

    output = np.zeros_like(im)

    output[:,:,0] = gamma * (im[:,:,0] - avgLuminance)
    output[:,:,1] = gamma * (im[:,:,1] - avgLuminance)
    output[:,:,2] = gamma * (im[:,:,2] - avgLuminance)

    return output