import numpy as np
def ColorContrastEnhancement(sceneRadiance):
    bgrAvg = np.zeros(3)
    sceneRadiance = np.float16(sceneRadiance)
    for k in range(0, 3):
        bgrAvg[k] = np.mean(sceneRadiance[:,:,k])
    # print('bgrAvg',bgrAvg)
    for k in range(1, 3):
        sceneRadiance[:, :, k] = (bgrAvg[0]/bgrAvg[k])*sceneRadiance[:,:,k]
    for i in range(0, 3):
        for j in range(0, sceneRadiance.shape[0]):
            for k in range(0, sceneRadiance.shape[1]):
                if sceneRadiance[j, k, i] > 255:
                    sceneRadiance[j, k, i] = 255
                if sceneRadiance[j, k, i] < 0:
                    sceneRadiance[j, k, i] = 0
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance



