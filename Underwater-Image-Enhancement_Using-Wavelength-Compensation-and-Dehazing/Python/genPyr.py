from pyr_reduce import pyr_reduce
import numpy as np
from pyr_expand import pyr_expand

def genPyr(img, t, level):
    pyr = [0 for i in range(level)]
    pyr[0] = img
    for p in range(1, level):
        pyr[p] = pyr_reduce(pyr[p-1])

    if t == 'gauss':
        return pyr
    osz = [0, 0, 1]
    for p in range(level-2, -1, -1):
        osz[0], osz[1] = (pyr[p+1].shape)
        for i in range(3):
            osz[i] = osz[i] * 2 - 1
        pyr[p] = pyr[p][0:osz[0], 0:osz[1]]

    for p in range(level-1):
        try:
            pyr[p] = pyr[p] - pyr_expand(pyr[p+1])
        except:
            pyr[p] = pyr[p] - pyr_expand(pyr[p+1])[:,:,0]

    return pyr 
    