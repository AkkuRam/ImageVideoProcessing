import math
import cv2
from PIL import Image
import matplotlib.pyplot as plot
import numpy as np

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

def distance_computation(u, v, height, width):
    return np.sqrt((u - height//2)**2 + ((v - width//2)**2))

def low_pass_filter(d0, n1, n2, n):
    k1,k2 = np.meshgrid(np.arange(-round(n2/2)+1, math.floor(n2/2)+1), np.arange(-round(n1/2)+1, math.floor(n1/2)+1))
    d = np.sqrt(k1**2 + k2**2)
    h = 1 / (1 + (d / d0)**(2*n))
    return h

def high_pass_filter():
    pass

def human_perception():
    # Original images I am using for low and high pass filters
    img_1 = cv2.imread('images-project2/face11.jpg', 0)
    img_2 = cv2.cvtColor(cv2.imread('images-project2/face12.jpg'), cv2.COLOR_BGR2RGB)

    img_low_pass = low_pass_filter(20,img_1.shape[0],img_1.shape[1],1)
    fft_img = np.fft.fftshift(np.fft.fft2(img_1)) * img_low_pass
    img_1_restored = np.fft.ifft2(np.fft.ifftshift(fft_img)).clip(0, 255).astype(np.uint8)

    display_image(images=[img_1_restored, img_low_pass], figsize=(10,8), rows=2, cols = 2, titles=['Original image', 'Low Pass'])




def main():
    human_perception()

main()