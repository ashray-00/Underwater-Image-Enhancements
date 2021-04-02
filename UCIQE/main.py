import cv2
from skimage import color
import numpy as np

def calculate_UCIQE(image, c1=0.4680, c2=0.2745, c3=0.2576):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    # lab = color.rgb2lab(rgb)
    print(lab[0,0,0])
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
    image_file = input("Enter image locattion")
    image = cv2.imread(image_file)
    uciqe = calculate_UCIQE(image)
    print(uciqe)


if __name__ == "__main__":
    main()