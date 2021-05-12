import os
import datetime
import numpy as np
import cv2
import natsort
from Blue_green_channels_dehazing_and_red_channel_correction.DetermineDepth import determineDepth
from Blue_green_channels_dehazing_and_red_channel_correction.TransmissionEstimation import getTransmission
from Blue_green_channels_dehazing_and_red_channel_correction.getAdaptiveExposureMap import AdaptiveExposureMap
from Blue_green_channels_dehazing_and_red_channel_correction.getAdaptiveSceneRadiance import AdaptiveSceneRadiance
from Blue_green_channels_dehazing_and_red_channel_correction.getAtomsphericLight import getAtomsphericLight
from Blue_green_channels_dehazing_and_red_channel_correction.refinedTransmission import refinedtransmission

from Blue_green_channels_dehazing_and_red_channel_correction.sceneRadianceGb import sceneRadianceGB
from Blue_green_channels_dehazing_and_red_channel_correction.sceneRadianceR import sceneradiance


def Blue_green_channels_dehazing_and_red_channel_correction(img):
    np.seterr(over='ignore')
    img = (img - img.min()) / (img.max() - img.min()) * 255
    blockSize = 9
    largestDiff = determineDepth(img, blockSize)
    AtomsphericLight, AtomsphericLightGB, AtomsphericLightRGB = getAtomsphericLight(largestDiff, img)
    
    transmission = getTransmission(img, AtomsphericLightRGB, blockSize)

    transmission = refinedtransmission(transmission, img)

    sceneRadiance_GB = sceneRadianceGB(img, transmission, AtomsphericLightRGB)

    sceneRadiance = sceneradiance(img, sceneRadiance_GB)

    S_x = AdaptiveExposureMap(img, sceneRadiance, Lambda=0.3, blockSize=blockSize)
        
    sceneRadiance = AdaptiveSceneRadiance(sceneRadiance, S_x)

    return sceneRadiance


