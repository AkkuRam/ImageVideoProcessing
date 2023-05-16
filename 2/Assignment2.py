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


def human_perception():
    img_low_pass = cv2.cvtColor(cv2.imread('images-project2/face11.jpg'), cv2.COLOR_BGR2RGB)
    img_high_pass = cv2.cvtColor(cv2.imread('images-project2/face12.jpg'), cv2.COLOR_BGR2RGB)
    display_image(images=[img_low_pass, img_high_pass], figsize=(10,8), rows=1, cols = 2, titles=['Original image', 'Original image'])




def main():
    human_perception()

main()