import numpy as np
import cv2
import csv
import math
from PIL import Image
import sys
from skimage import color, filters
import imageio
import os
import math
import natsort
from scipy import ndimage

def mu_a(x, alpha_L=0.1, alpha_R=0.1):

   # sort pixels by intensity - for clipping
   x = sorted(x)

   # get number of pixels
   K = len(x)

   # calculate T alpha L and T alpha R
   T_a_L = math.ceil(alpha_L*K)
   T_a_R = math.floor(alpha_R*K)

   # calculate mu_alpha weight
   weight = (1/(K-T_a_L-T_a_R))

   # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
   s   = int(T_a_L+1)
   e   = int(K-T_a_R)
   val = sum(x[s:e])
   val = weight*val
   return val

def s_a(x, mu):
   val = 0
   for pixel in x:
      val += math.pow((pixel-mu), 2)
   return val/len(x)

def _uicm(x):

   R = x[:,:,0].flatten()
   G = x[:,:,1].flatten()
   B = x[:,:,2].flatten()

   RG = R-G
   YB = ((R+G)/2)-B

   mu_a_RG = mu_a(RG)
   mu_a_YB = mu_a(YB)

   s_a_RG = s_a(RG, mu_a_RG)
   s_a_YB = s_a(YB, mu_a_YB)

   l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
   r = math.sqrt(s_a_RG+s_a_YB)

   return (-0.0268*l)+(0.1586*r)

def sobel(x):
   dx = ndimage.sobel(x,0)
   dy = ndimage.sobel(x,1)
   mag = np.hypot(dx, dy)
   mag *= 255.0 / np.max(mag) 
   return mag

'''
   Enhancement measure estimation
   x.shape[0] = height
   x.shape[1] = width
'''
def eme(x, window_size):

   # if 4 blocks, then 2x2...etc.
   k1 = x.shape[1]//window_size
   k2 = x.shape[0]//window_size

   # weight
   w = 2./(k1*k2)

   blocksize_x = window_size
   blocksize_y = window_size

   # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
   x = x[:blocksize_y*k2, :blocksize_x*k1]

   val = 0
   for l in range(k1):
      for k in range(k2):
         block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
         max_ = np.max(block)
         min_ = np.min(block)

         # bound checks, can't do log(0)
         if min_ == 0.0: val += 0
         elif max_ == 0.0: val += 0
         else: val += math.log(max_/min_)

   return w*val

'''
   Underwater Image Sharpness Measure
'''
def _uism(x):

   # get image channels
   R = x[:,:,0]
   G = x[:,:,1]
   B = x[:,:,2]

   # first apply Sobel edge detector to each RGB component
   Rs = sobel(R)
   Gs = sobel(G)
   Bs = sobel(B)

   # multiply the edges detected for each channel by the channel itself
   R_edge_map = np.multiply(Rs, R)
   G_edge_map = np.multiply(Gs, G)
   B_edge_map = np.multiply(Bs, B)

   # get eme for each channel
   r_eme = eme(R_edge_map, 10)
   g_eme = eme(G_edge_map, 10)
   b_eme = eme(B_edge_map, 10)

   # coefficients
   lambda_r = 0.299
   lambda_g = 0.587
   lambda_b = 0.144

   return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
   return mu-x

def plip_theta(g1, g2, k):
   g1 = plip_g(g1)
   g2 = plip_g(g2)
   return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
   g1 = plip_g(g1)
   g2 = plip_g(g2)
   return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
   g = plip_g(g)
   return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
   return plip_phiInverse(plip_phi(g1) * plip_phi(g2))

def plip_phiInverse(g):
   plip_lambda = 1026.0
   plip_beta   = 1.0
   return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta))

def plip_phi(g):
   plip_lambda = 1026.0
   plip_beta   = 1.0
   return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)


'''
   Underwater image contrast measure
'''
def _uiconm(x, window_size):

   '''
      PLIP definitions
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5609219
   '''
   plip_lambda = 1026.0
   plip_gamma  = 1026.0
   plip_beta   = 1.0
   plip_mu     = 1026.0
   plip_k      = 1026.0
   
   # if 4 blocks, then 2x2...etc.
   k1 = x.shape[1]//window_size
   k2 = x.shape[0]//window_size

   # weight
   w = -1./(k1*k2)

   blocksize_x = window_size
   blocksize_y = window_size

   # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
   x = x[:blocksize_y*k2, :blocksize_x*k1]

   # entropy scale - higher helps with randomness
   alpha = 1

   val = 0
   for l in range(k1):
      for k in range(k2):
         block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
         max_ = np.max(block)
         min_ = np.min(block)

         top = max_-min_
         bot = max_+min_


         if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
         else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
         
         #try: val += plip_multiplication((top/bot),math.log(top/bot))


   #return plip_diag(w, val, plip_gamma)
   return w*val

def calc_uiqm(x):
   x = x.astype(np.float32)
   c1 = 0.0282
   c2 = 0.2953
   c3 = 3.5753
   uicm   = _uicm(x)
   uism   = _uism(x)
   uiconm = _uiconm(x, 10)
   uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
   return uiqm

def main():
    file_path = "./OutputImages/"
    output_path = "./Metric_Output/"

    folders = os.listdir(file_path)
    methods = ['Images']
    images = []
    folders = natsort.natsorted(folders)
    flag = False
    flag1 = False
    count = 0
    for i in range(len(folders)):
        count += 1
        method = folders[i]
        methods.append(str(method))
        path = file_path + method + "/"
        files = os.listdir(path)
        files = natsort.natsorted(files)
        if (flag == False):
            UIQM = [[0 for m in range(len(folders) + 1)] for n in range(len(files))]
            flag = True
        for j in range(len(files)):
            file = files[j]
            if (i == 0):
                UIQM[j][i] = str(file)
            image = cv2.imread(path + file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = imageio.imread(path + file)
            image = image.astype(np.float32)
            uiqm = calc_uiqm(image)
            UIQM[j][i+1] = uiqm
        flag1 = True
    with open(output_path + 'UIQM.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(methods)
        for i in range(len(UIQM)):
            writer.writerow(UIQM[i])

if __name__ == "__main__":
    main()
