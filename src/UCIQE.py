import cv2
import os
from skimage import color
import natsort
import csv
import numpy as np

def calc_uciqe(img_bgr):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

                                     # According to training result mentioned in the paper:
    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)
    return quality_val

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
            uciqe = calc_uciqe(image)
            UCIQUE[j][i+1] = uciqe
        flag1 = True
    with open(output_path + 'UCIQUE.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(methods)
        for i in range(len(UCIQUE)):
            writer.writerow(UCIQUE[i])
        


if __name__ == "__main__":
    main()
