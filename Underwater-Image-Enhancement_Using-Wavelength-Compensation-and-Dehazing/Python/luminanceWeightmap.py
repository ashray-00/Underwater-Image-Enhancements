import numpy as np

def luminanceWeightmap(im):

    L = np.mean(im, axis=2)

    output = np.sqrt(1/3 * np.square(im[:,:,0] - L) + np.square(im[:,:,1] - L) + np.square(im[:,:,2] - L))

    return output