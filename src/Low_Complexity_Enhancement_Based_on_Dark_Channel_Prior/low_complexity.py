import os
import numpy as np
import cv2
import natsort
import matplotlib.pyplot as plt


from Low_Complexity_Enhancement_Based_on_Dark_Channel_Prior.TransmissionMap import TransmissionComposition
from Low_Complexity_Enhancement_Based_on_Dark_Channel_Prior.getAtomsphericLight import getAtomsphericLight
from Low_Complexity_Enhancement_Based_on_Dark_Channel_Prior.getColorContrastEnhancement import ColorContrastEnhancement
from Low_Complexity_Enhancement_Based_on_Dark_Channel_Prior.getRGBDarkChannel import getDarkChannel
from Low_Complexity_Enhancement_Based_on_Dark_Channel_Prior.getSceneRadiance import SceneRadiance


from Low_Complexity_Enhancement_Based_on_Dark_Channel_Prior.getTransmissionEstimation import getTransmissionMap

def low_complexity(img):
    np.seterr(over='ignore')
    blockSize = 9
    imgGray = getDarkChannel(img, blockSize)
    AtomsphericLight = getAtomsphericLight(imgGray, img, meanMode=True, percent=0.01)

    transmission = getTransmissionMap(img, AtomsphericLight, blockSize)
    sceneRadiance = SceneRadiance(img, AtomsphericLight, transmission)
    sceneRadiance = ColorContrastEnhancement(sceneRadiance)

    return sceneRadiance


