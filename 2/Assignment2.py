import math
import cv2
from PIL import Image
import matplotlib.pyplot as plot
import numpy as np
import warnings

import scipy

warnings.filterwarnings('ignore')
np.seterr(divide="ignore", invalid="ignore")

# This is my function to display images, where its first parameters can take in multiple images
# The second parameter is a default size of the frame of displayed images, which is changeable
# Then you can define the rows and columns for the subplots in the frame
# The titles parameter takes in multiple title names for the given images
# The mode parameter specifies whether you want it as an image, histogram or lineplot
# Lastly for the axis you can choose to have it on or off
def display_image(images, figsize=(20, 7), rows=None, cols=None, titles=None, mode='image', axis='off'):

    fig, axes = plot.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            if mode == 'image':
                ax.imshow(images[i], cmap='gray')
            elif mode == 'histogram':
                ax.hist(images[i].ravel(), bins=256, color='gray')
            elif mode == 'lineplot':
                ax.plot(images[i])
            if titles is not None:
                ax.set_title(titles[i])
        if axis =='on':
            ax.axis('on')
        else:
            ax.axis('off')

    plot.show()

def distance_computation(height, width, u, v):
    return np.sqrt((u - height//2)**2 + (v - width//2)**2)

def butterworth_low_pass(image, d0, n):
    h = np.zeros(image.shape[:2])
    height, width = image.shape[:2]
    for u in range(height):
        for v in range(width):
            h[u,v] = (1/(1 + (distance_computation(height, width, u, v)/d0)** (2*n)))
    return h

def butterworth_high_pass(image, d0, n):
    h = np.zeros(image.shape[:2])
    height, width = image.shape[:2]
    for u in range(height):
        for v in range(width):
            h[u,v] = (1/(1 + (d0/distance_computation(height, width, u, v))** (2*n)))
    return h


def human_perception():
    # Original images I am using for low and high pass filters
    img_1 = cv2.imread('images-project2/face11.jpg', 0)
    img_2 = cv2.imread('images-project2/face12.jpg', 0)

    img_low_pass = butterworth_low_pass(img_1, 20, 2)
    fft_img_1 = np.fft.fftshift(np.fft.fft2(img_1)) * img_low_pass
    img_1_restored = np.fft.ifft2(np.fft.ifftshift(fft_img_1)).clip(0, 255).astype(np.uint8)

    img_high_pass = butterworth_high_pass(img_2, 20, 2)
    fft_img_2 = np.fft.fftshift(np.fft.fft2(img_2)) * img_high_pass
    img_2_restored = np.fft.ifft2(np.fft.ifftshift(fft_img_2)).clip(0, 255).astype(np.uint8)

    combined_img = cv2.add(img_1_restored, img_2_restored)
  

    display_image(images=[img_1_restored, img_2_restored, combined_img], figsize=(10,5), rows=1, cols = 3, titles=['Low pass image', 'High pass image', 'Combined image'])

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.T, norm='ortho' ).T, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')

def blockwise_dct(img):
    imsize = img.shape
    dct = np.zeros(imsize)

    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            dct[i:i+8,j:j+8] = dct2( img[i:i+8,j:j+8] )

def zigzag():
    pass

def watermark():
    pos = 128
    k = 10
    kirby_img = cv2.imread('images-project2/kirby.jpg')
    kirby_gray = cv2.cvtColor(kirby_img, cv2.COLOR_BGR2GRAY)
    dct_block = blockwise_dct(kirby_gray)

    display_image(images=[kirby_gray], figsize=(10,5), rows=1, cols = 3, titles=['Original image', 'Watermark image', 'Difference image'])


def main():
    # human_perception()
    watermark()

main()