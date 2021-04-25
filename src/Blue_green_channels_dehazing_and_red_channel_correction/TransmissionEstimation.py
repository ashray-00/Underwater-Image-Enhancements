import numpy as np
import math

def getTransmission(normI,AtomsphericLight ,w):
    M, N, C = normI.shape #M are the rows, N are the columns, C is the bgr channel
    B = AtomsphericLight
    padwidth = math.floor(w/2)
    padded = np.pad(normI/B, ((padwidth, padwidth), (padwidth, padwidth),(0,0)), 'constant')
    transmission = np.zeros((M,N,2))
    for y, x in np.ndindex(M, N):
        transmission[y,x,0] = 1 - np.min(padded[y : y+w , x : x+w , 0])
        transmission[y,x,1] = 1 - np.min(padded[y : y+w , x : x+w , 1])
    return transmission