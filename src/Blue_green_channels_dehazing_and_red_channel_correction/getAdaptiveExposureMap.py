import numpy as np
import cv2

from Blue_green_channels_dehazing_and_red_channel_correction.GuidedFilter import GuidedFilter
from Blue_green_channels_dehazing_and_red_channel_correction.guidedfilter_He import guided_filter_he

np.seterr(over='ignore')
np.seterr(invalid ='ignore')
np.seterr(all ='ignore')


def AdaptiveExposureMap(img, sceneRadiance, Lambda, blockSize):

    minValue = 10 ** -2
    img = np.uint8(img)
    sceneRadiance = np.uint8(sceneRadiance)

    YjCrCb = cv2.cvtColor(sceneRadiance, cv2.COLOR_BGR2YCrCb)
    YiCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    normYjCrCb = (YjCrCb - YjCrCb.min()) / (YjCrCb.max() - YjCrCb.min())
    normYiCrCb = (YiCrCb - YiCrCb.min()) / (YiCrCb.max() - YiCrCb.min())
    Yi = normYiCrCb[:, :, 0]
    Yj = normYjCrCb[:, :, 0]
    Yi = np.clip(Yi, minValue,1)
    Yj = np.clip(Yj, minValue,1)

    S = (Yj * Yi + Lambda * Yi ** 2) / (Yj ** 2 + Lambda * Yi ** 2)

    gimfiltR = 50 
    eps = 10 ** -3  

    guided_filter = GuidedFilter(YiCrCb, gimfiltR, eps)

    refinedS = guided_filter.filter(S)

    S_three = np.zeros(img.shape)
    S_three[:, :, 0] = S_three[:, :, 1] = S_three[:, :, 2] = refinedS

    return S_three