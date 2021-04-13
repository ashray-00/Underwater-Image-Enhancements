import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from skimage import color, exposure, transform, img_as_float32, img_as_uint, img_as_ubyte
from Wavelength_Compensation_and_Dehazing.whiteBalance import whiteBalance
from Wavelength_Compensation_and_Dehazing.enhanceContrast import enhanceContrast
from Wavelength_Compensation_and_Dehazing.luminanceWeightmap import luminanceWeightmap
from Wavelength_Compensation_and_Dehazing.chromaticWeightmap import chromaticWeightmap
from Wavelength_Compensation_and_Dehazing.saliencyWeightmap import saliencyWeightmap
from Wavelength_Compensation_and_Dehazing.pyrReconstruct import pyrReconstruct
from Wavelength_Compensation_and_Dehazing.genPyr import genPyr

def wavelength_compensation_and_dehazing(im):
    np.seterr(divide='ignore', invalid='ignore')

    im = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255.0)
    im = img_as_float32(im)

    firstInput = whiteBalance(im)

    secondInput = enhanceContrast(im)

    luminanceWeightmap1 = luminanceWeightmap(firstInput)
    chromaticWeightmap1 = chromaticWeightmap(firstInput)
    saliencyWeightmap1 = saliencyWeightmap(firstInput)
    resultedWeightmap1 = luminanceWeightmap1 * chromaticWeightmap1 * saliencyWeightmap1

    luminanceWeightmap2 = luminanceWeightmap(secondInput)
    chromaticWeightmap2 = chromaticWeightmap(secondInput)
    saliencyWeightmap2 = saliencyWeightmap(secondInput)

    resultedWeightmap2 = luminanceWeightmap2 * chromaticWeightmap2 * saliencyWeightmap2

    normalizedWeightmap1 = resultedWeightmap1 / (resultedWeightmap1 + resultedWeightmap2)
    normalizedWeightmap2 = resultedWeightmap2 / (resultedWeightmap1 + resultedWeightmap2)

    gaussianPyramid1 = genPyr(normalizedWeightmap1, 'gauss', 5)
    gaussianPyramid2 = genPyr(normalizedWeightmap2, 'gauss', 5)

    fusedPyramid = []
    for i in range(5):
        tempImg = []
        for j in range(im.shape[2]):
            laplacianPyramid1 = genPyr(firstInput[:,:,j], 'laplace', 5)
            laplacianPyramid2 = genPyr(secondInput[:,:,j], 'laplace', 5)
            rowSize = np.min([laplacianPyramid1[i].shape[0], laplacianPyramid2[i].shape[0], gaussianPyramid1[i].shape[0], gaussianPyramid2[i].shape[0]])
            columnSize = np.min([laplacianPyramid1[i].shape[1], laplacianPyramid2[i].shape[1], gaussianPyramid1[i].shape[1], gaussianPyramid2[i].shape[1]])
            tempImg.append(laplacianPyramid1[i][0:rowSize, 0:columnSize] * gaussianPyramid1[i][0:rowSize, 0:columnSize] + laplacianPyramid2[i][0:rowSize, 0:columnSize] * gaussianPyramid2[i][0:rowSize, 0:columnSize])
        fusedPyramid.append([(np.array(tempImg)).transpose(1,2,0)])
    
    for i in range(5):
        fusedPyramid[i] = fusedPyramid[i][0]

    enhancedImage = pyrReconstruct(fusedPyramid)
    if (np.max(enhancedImage) < 2.0):
        enhancedImage /= np.max(enhancedImage)
    
    output = img_as_ubyte(enhancedImage)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

if __name__ == "__main__":
    im = cv2.imread(sys.argv[1])
    output = wavelength_compensation_and_dehazing(im)