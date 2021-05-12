import sys
import cv2
from PIL import Image
import numpy as np
from skimage import color, exposure, transform, img_as_float32, img_as_uint, img_as_ubyte
from scipy.ndimage import gaussian_filter, convolve

def saliency_detection(image):
    """
    Frequency-tuned Salient Region Detection
    Radhakrishna Achanta, Sheila Hemami, Francisco Estrada, Sabine Susstrunk
    Abstract
    Detection of visually salient image regions is useful for applications like
    object segmentation, adaptive compression, and object recognition. In this
    paper, we introduce a method for salient region detection that outputs full
    resolution saliency maps with well-defined boundaries of salient objects.
    These boundaries are preserved by retaining substantially more frequency
    content from the original image than other existing techniques. Our method
    exploits features of color and luminance, is simple to implement, and is
    computationally efficient. We compare our algorithm to five
    state-of-the-art salient region detection methods with a frequency domain
    analysis, ground truth, and a salient object segmentation application. Our
    method outperforms the five algorithms both on the ground truth evaluation
    and on the segmentation task by achieving both higher precision and better
    recall.
    Reference and PDF
    R. Achanta, S. Hemami, F. Estrada and S. Susstrunk, Frequency-tuned Salient
    Region Detection, IEEE International Conference on Computer Vision and
    Pattern Recognition (CVPR), 2009.
    URL: http://ivrg.epfl.ch/supplementary_material/RK_CVPR09/
    Args:
        image: numpy array
          RGB image as an array
    Returns:
        sm: numpy array
           Normalised salicency map
    """

    if len(image.shape) != 3:
        #print 'Warning processing non standard image'
        image = color.gray2rgb(image)

    # Convert to CIE Lab colour space

    image = gaussian_filter(image, sigma=3)
    image = color.rgb2lab(image)

    # Get each channel

    l = convert_rgb(image, 0)[:,:,0]
    a = convert_rgb(image, 1)[:,:,1]
    b = convert_rgb(image, 2)[:,:,2]

    # LAB image average

    lm = np.mean(l)
    am = np.mean(a)
    bm = np.mean(b)

    # Compute the saliency map

    sm = np.sqrt(np.square(l - lm) + np.square(a - am) + np.square(b - bm))

    # Normalise saliency map

    sm *= 255 / sm.max()
    return sm

def convert_rgb(image, idim):
    z = np.zeros(image.shape)
    if idim != 0 :
        z[:,:,0]=80
    z[:,:,idim] = image[:,:,idim]
    z = color.lab2rgb(z)
    return(z)

def pyramid_reconstruct(img):
    pyramid = img
    level = len(pyramid)
    for i in range(level-1, 0, -1):
        m, n = pyramid[i - 1].shape
        pyramid[i - 1] = pyramid[i - 1] + np.resize(pyramid[i], (m,n))
    return pyramid[0]/255

def laplacian_pyramid(img, level):
    out = [0 for i in range(level)]
    out[0] = img
    temp_img = img
    for i in range(1, level):
        temp_img = temp_img[0::2,0::2]
        out[i] = temp_img
    
    for i in range(level - 1):
        m, n = out[i].shape
        out[i] = out[i] - np.resize(out[i+1], (m,n))
    
    return out

def gaussian_pyramid(img, level):
    out = [0 for i in range(level)]
    h = 1/16 * np.array([[1, 4, 6, 4, 1]])
    filt = np.matmul(np.transpose(h), h)
    out[0] = convolve(img, filt, mode='nearest')
    temp_img = img
    for i in range(1, level):
        temp_img = temp_img[0::2,0::2]
        out[i] = convolve(temp_img, filt, mode='nearest')
    
    return out
    

def color_balance_and_fusion(input_img):
    np.seterr(divide='ignore', invalid='ignore')
    rgbImage = input_img
    rgbImage = (cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)/255.0)
    rgbImage = img_as_float32(rgbImage)
    greyImage = color.rgb2gray(rgbImage)

    #Since cv2 representss image in bgr format
    Ir = rgbImage[:,:,0]
    Ig = rgbImage[:,:,1]
    Ib = rgbImage[:,:,2]

    Ir_mean = np.mean(Ir)
    Ig_mean = np.mean(Ig)
    Ib_mean = np.mean(Ib)

    alpha = 1
    Irc = Ir + alpha * (Ig_mean - Ir_mean) * (1 - Ir) * Ig
    
    alpha = 0
    Ibc = Ib + alpha * (Ig_mean - Ib_mean) * (1 - Ib) * Ig

    I = np.dstack((Irc, Ig, Ibc))

    I = img_as_uint(I)

    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.95)
    Iwb = wb.balanceWhite(I)
    Iwb = img_as_float32(Iwb)

    Igamma = exposure.adjust_gamma(Iwb, 1, 1)
    Igamma /= np.max(Igamma)

    sigma = 20
    Igauss = Iwb
    N = 30
    for i in range(N):
        Igauss = gaussian_filter(Igauss, sigma)
        Igauss = np.minimum(Iwb, Igauss)
    
    gain = 1
    Norm = (Iwb - gain * Igauss)
    for i in range(3):
        Norm[:,:,i] = exposure.equalize_hist(Norm[:,:,i])
    Isharp = (Iwb + Norm)/2

    Isharp_lab = color.rgb2lab(Isharp/255)
    Igamma_lab = color.rgb2lab(Igamma/255)

    R1 = convert_rgb(Isharp_lab, 0)/255
    R1 /= np.max(R1)
    test = R1
    WC1 = np.sqrt((np.square(Isharp[:,:,0] - R1[:,:,0]) + np.square(Isharp[:,:,1] - R1[:,:,0]) + np.square(Isharp[:,:,2] - R1[:,:,0]))/3)
    WS1 = saliency_detection(Isharp)
    WS1 = WS1/np.max(WS1)

    WSAT1 = np.sqrt(1/3*(np.square(Isharp[:,:,0]-R1[:,:,0])+np.square(Isharp[:,:,1]-R1[:,:,0])+np.square(Isharp[:,:,2]-R1[:,:,0])))
    
    R2 = convert_rgb(Igamma_lab, 0)/255
    R2 /= np.max(R2)
    test1 = R2
    WC2 = np.sqrt(((np.square(Igamma[:,:,0] - R2[:,:,0]) + np.square(Igamma[:,:,1] - R2[:,:,0]) + np.square(Igamma[:,:,2] - R2[:,:,0]))/3))
    WS2 = saliency_detection(Igamma)
    WS2 = WS2/np.max(WS2)

    WSAT2 = np.sqrt(((np.square(Igamma[:,:,0] - R1[:,:,0]) + np.square(Igamma[:,:,1] - R1[:,:,0]) + np.square(Igamma[:,:,2] - R1[:,:,0]))/3))
    
    W1 = (WC1 + WS1 + WSAT1 + 0.1)/(WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2)
    W2 = (WC2 + WS2 + WSAT2 + 0.1)/(WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2 + 0.2)

    naive = W1[...,np.newaxis] * Isharp + W2[..., np.newaxis] * Igamma
    
    img1 = Isharp
    img2 = Igamma

    level = 15

    Weight1 = gaussian_pyramid(W1, level)
    Weight2 = gaussian_pyramid(W2, level)

    R1 = laplacian_pyramid(Isharp[:,:,0], level)
    G1 = laplacian_pyramid(Isharp[:,:,1], level)
    B1 = laplacian_pyramid(Isharp[:,:,2], level)

    R2 = laplacian_pyramid(Igamma[:,:,0], level)
    G2 = laplacian_pyramid(Igamma[:,:,1], level)
    B2 = laplacian_pyramid(Igamma[:,:,2], level)

    Rr = np.zeros_like(Weight1)
    Rg = np.zeros_like(Weight1)
    Rb = np.zeros_like(Weight1)

    for k in range(level):
        Rr[k] = Weight1[k] * R1[k] + Weight2[k] * R2[k]
        Rg[k] = Weight1[k] * G1[k] + Weight2[k] * G2[k]
        Rb[k] = Weight1[k] * B1[k] + Weight2[k] * B2[k]

    R = pyramid_reconstruct(Rr)
    G = pyramid_reconstruct(Rg)
    B = pyramid_reconstruct(Rb)

    fusion = np.dstack((R, G, B))
    fusion /= np.max(fusion)
    output = img_as_ubyte(fusion)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def main():
    input_img = cv2.imread(sys.argv[1])
    output = color_balance_and_fusion(input_img)
    cv2.imwrite("a3.png", output)

if __name__ == "__main__":
    main()
