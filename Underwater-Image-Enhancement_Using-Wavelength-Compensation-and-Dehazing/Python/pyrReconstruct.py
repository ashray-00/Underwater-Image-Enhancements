from pyr_expand import pyr_expand
def pyrReconstruct(pyr):
    for p in range(len(pyr)-2, -1, -1):
        pyr[p] = pyr[p] + pyr_expand(pyr[p+1])
    
    return pyr[0]