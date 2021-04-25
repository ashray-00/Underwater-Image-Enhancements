import numpy as np
from Blue_green_channels_dehazing_and_red_channel_correction.GuidedFilter import GuidedFilter


def  refinedtransmission(transmission, img):
    gimfiltR = 50  # 引导滤波时半径的大小
    eps = 10 ** -3  # 引导滤波时epsilon的值

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmission[:,:,0] = guided_filter.filter(transmission[:,:,0])
    transmission[:,:,1] = guided_filter.filter(transmission[:,:,1])
    transmission = np.clip(transmission, 0.1, 0.9)


    return transmission

