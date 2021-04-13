import cv2
import os
from skimage import color
import natsort
import csv
import numpy as np

def calculate_UCIQE(image, c1=0.4680, c2=0.2745, c3=0.2576):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    # lab = color.rgb2lab(rgb)
    l = lab[:,:,0]/255
    a = lab[:,:,1]/255
    b = lab[:,:,2]/255

    chroma = np.sqrt(np.square(a) + np.square(b))

    u_c = np.mean(chroma)

    sigma_c = np.sqrt(np.mean(np.abs(1 - np.square(u_c/chroma))))

    saturation = chroma / np.sqrt(np.square(chroma) + np.square(l))

    u_s = np.mean(saturation)

    contrast_l = np.max(l) - np.min(l)

    UCIQE = c1 * sigma_c + c2 * contrast_l + c3 * u_s
    return UCIQE

def main():
    file_path = "./OutputImages/"
    output_path = "./Metric_Output/"

    folders = os.listdir(file_path)
    methods = ['Images']
    images = []
    folders = natsort.natsorted(folders)
    flag = False
    flag1 = False
    for i in range(len(folders)):
        method = folders[i]
        methods.append(str(method))
        path = file_path + method + "/"
        files = os.listdir(path)
        files = natsort.natsorted(files)
        if (flag == False):
            # UCIQUE = np.zeros((int(len(files)), int(len(folders) + 1)))
            UCIQUE = [[0 for m in range(len(folders) + 1)] for n in range(len(files))]
            flag = True
        for j in range(len(files)):
            file = files[j]
            if (i == 0):
                UCIQUE[j][i] = str(file)
            image = cv2.imread(path + file)
            uciqe = calculate_UCIQE(image)
            UCIQUE[j][i+1] = uciqe
        flag1 = True
    with open(output_path + 'UCIQUE.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(methods)
        for i in range(len(UCIQUE)):
            writer.writerow(UCIQUE[i])
        


if __name__ == "__main__":
    main()