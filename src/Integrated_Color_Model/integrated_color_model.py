import os
import numpy as np
import cv2
import natsort
import xlwt
from Integrated_Color_Model.global_histogram_stretching import stretching
from Integrated_Color_Model.hsvStretching import HSVStretching
from Integrated_Color_Model.sceneRadiance import sceneRadianceRGB


def integrated_color_model(img):
    np.seterr(over='ignore')
    img = stretching(img)
    sceneRadiance = sceneRadianceRGB(img)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)

    return sceneRadiance
