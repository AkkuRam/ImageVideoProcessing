import cv2
from PIL import Image
import matplotlib.pyplot as plot
import numpy as np

np.seterr(divide="ignore", invalid="ignore")

# Flamingo and Deer image in BGR formats
flamingo_img = cv2.imread('images/flamingo.jpg')
deer_img = cv2.imread('images/deer.jpg')

# Flamingo and Deer image converted to RGB formats
flamingo_rgb = cv2.cvtColor(flamingo_img, cv2.COLOR_BGR2RGB)
deer_rgb = cv2.cvtColor(deer_img, cv2.COLOR_BGR2RGB)

# Beach and Diag2 image in BGR formats 
beach_img = cv2.imread('images/beach.jpg')

# Birdie image in BGR format
birdie_img = cv2.imread('images/birdie.jpg')

# Sheep image in BGR format
sheep_img = cv2.imread('images/sheep.jpg')

# Grid image in BGR format
palette_img = cv2.imread('images/palette.jpg')

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

# THe parameters are each channel r,g,b and h,s,v extracted from both flamingo and deer images for display
def rgb_to_hsv(fr, fg, fb, fh, fs, fv, dr, dg, db, dh, ds, dv):

    # Here I create a plot of 2 rows and 6 cols to display all the 6 individual channels for the original images in RGB
    # and for the images converted to HSV
    channel_images = [fr, fg, fb, fh, fs, fv, dr, dg, db, dh, ds, dv]
    channel_images_titles = ['R channel', 'G channel', 'B channel', 'R channel', 'G channel', 'B channel', 'H channel', 'S channel', 'V channel', 'H channel', 'S channel', 'V channel']
    display_image(images=channel_images, rows=2, cols=6, titles=channel_images_titles)

    plot.show()

# This method does the histogram equalisation of the flamingo and deer images and displays them
def histogram_equalisation(fh, fs, fv, dh, ds, dv):

    # histogram equalization for for both images h,s,v component for flamingo
    fh_eq = cv2.equalizeHist(fh)
    fs_eq = cv2.equalizeHist(fs)
    fv_eq = cv2.equalizeHist(fv)

    # Here I merge the equalised channels back together
    fhsv_merge = cv2.merge((fh_eq, fs_eq, fv_eq))
    # I convert them back to RGB for display purposes
    fhsv_merge_eq = cv2.cvtColor(fhsv_merge, cv2.COLOR_HSV2RGB)
    # Here I merge the V equalised back witht he old H and S channel
    fhsv_merge_v = cv2.merge((fh, fs, fv_eq))
    # I convert them back to RGB for display purposes
    fhsv_merge_v_eq = cv2.cvtColor(fhsv_merge_v, cv2.COLOR_HSV2RGB)

    # histogram equalization for for both images h,s,v component for deer
    dh_eq = cv2.equalizeHist(dh)
    ds_eq = cv2.equalizeHist(ds)
    dv_eq = cv2.equalizeHist(dv)
    # Here I merge the equalised channels back together
    dhsv_merge = cv2.merge((dh_eq, ds_eq, dv_eq))
    # I convert them back to RGB for display purposes
    dhsv_merge_eq = cv2.cvtColor(dhsv_merge, cv2.COLOR_HSV2RGB)
    # Here I merge the V equalised back witht he old H and S channel
    dhsv_merge_v = cv2.merge((dh, ds, dv_eq))
    # I convert them back to RGB for display purposes
    dhsv_merge_v_eq = cv2.cvtColor(dhsv_merge_v, cv2.COLOR_HSV2RGB)

    # Here I display the color images for the histogram equalised images
    hsv_color_images_titles = ['HSV Equalised', 'HSV Equalised', 'V Equalised', 'V Equalised']
    hsv_color_images = [fhsv_merge_eq,  dhsv_merge_eq, fhsv_merge_v_eq, dhsv_merge_v_eq]
    display_image(images = hsv_color_images, figsize=(8, 5), rows=1, cols=4, titles=hsv_color_images_titles, mode='image')
    
    # Here I am displaying histograms consisting of the histogram equalsied channels merged
    merged_color_images_titles = ['HSV Equalised Flamingo', 'HSV Equalised Deer', 'V Equalised Flamingo', 'V Equalised Deer']
    merged_color_images = [fhsv_merge, dhsv_merge, fhsv_merge_v, dhsv_merge_v]
    display_image(images=merged_color_images, rows=1, cols=4, figsize=(10, 5), titles=merged_color_images_titles, mode='histogram')

    # Here I am displaying the histogram equalised channel images
    channel_color_images_titles = ['H Equalised Flamingo', 'S Equalised Flamingo', 'V Equalised Flamingo', 'H Equalised Deer', 'S Equalised Deer', 'V Equalised Deer']
    channel_color_images = [cv2.cvtColor(fh_eq, cv2.COLOR_GRAY2RGB), cv2.cvtColor(fs_eq, cv2.COLOR_GRAY2RGB), cv2.cvtColor(fv_eq, cv2.COLOR_GRAY2RGB), cv2.cvtColor(dh_eq, cv2.COLOR_GRAY2RGB), cv2.cvtColor(ds_eq, cv2.COLOR_GRAY2RGB), cv2.cvtColor(dv_eq, cv2.COLOR_GRAY2RGB)]   
    display_image(images=channel_color_images, rows=2, cols=3, figsize=(10, 5), titles=channel_color_images_titles)

    plot.show()


def color_spaces():
    # I convert it to HSV from RGB, as I want to extract the h,s,v channels
    flamingo_hsv = cv2.cvtColor(flamingo_rgb, cv2.COLOR_RGB2HSV)
    deer_hsv = cv2.cvtColor(deer_rgb, cv2.COLOR_RGB2HSV)

    # We extract the r,g,b values from both images for displaying later
    fr, fg, fb = cv2.split(flamingo_rgb)
    dr, dg, db = cv2.split(deer_rgb)

    # We extract the h,s,v values from both images for displaying later
    fh, fs, fv = cv2.split(flamingo_hsv)
    dh, ds, dv = cv2.split(deer_hsv)

    # I am converting each channel back into RGB for display purposes
    fh_rgb = cv2.cvtColor(fh, cv2.COLOR_GRAY2RGB)
    fs_rgb = cv2.cvtColor(fs, cv2.COLOR_GRAY2RGB)
    fv_rgb = cv2.cvtColor(fv, cv2.COLOR_GRAY2RGB)
    dh_rgb = cv2.cvtColor(dh, cv2.COLOR_GRAY2RGB)
    ds_rgb = cv2.cvtColor(ds, cv2.COLOR_GRAY2RGB)
    dv_rgb = cv2.cvtColor(dv, cv2.COLOR_GRAY2RGB)

    # This method is to display all channels of both images, r,g,b and h,s,v
    rgb_to_hsv(fr, fg, fb, dr, dg, db, fh_rgb, fs_rgb, fv_rgb, dh_rgb, ds_rgb, dv_rgb)
    # This method displays resulting histogram, histogram equalised channels and color images 
    histogram_equalisation(fh, fs, fv, dh, ds, dv)

# This is my kernel filter for 45 degrees
def diagonal_kernel(image):
    # 45-degree filter kernel
    kernel_45= np.array([[-1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 1]])
    
    # Apply the 45-degree filter using filter2D
    filter_45_image = cv2.filter2D(image, -1, kernel_45)
    return filter_45_image

# This is my kernel filter for 135 degrees
def inverse_diagonal_kernel(image):
    # 135-degree filter kernel
    kernel_135= np.array([[0, 0, 1],
                      [0, 0, 0],
                      [-1, 0, 0]])
    
    # Apply the 135-degree filter using filter2D
    filter_135_image = cv2.filter2D(image, -1, kernel_135)
    return filter_135_image


def edge_detection_filter():

    # Convert the image to grayscale
    beach_gray = cv2.cvtColor(beach_img, cv2.COLOR_BGR2GRAY)

    # Here I add salt and pepper noise to the image, then I apply the 45 degree edge detection filter
    threshold, noise_filter_45_beach = cv2.threshold(diagonal_kernel(salt_pepper_noise(beach_gray)), 50, 255, cv2.THRESH_BINARY)
    # Here I add gaussian noise to the image, then I apply the 45 degree edge detection filter
    threshold, gauss_noise_filter_45_beach = cv2.threshold(diagonal_kernel(gaussian_noise(beach_gray)), 50, 255, cv2.THRESH_BINARY)

    # Here I add salt and pepper noise to the image, then I apply the 135 degree edge detection filter
    threshold, noise_filter_135_beach = cv2.threshold(inverse_diagonal_kernel(salt_pepper_noise(beach_gray)), 50, 255, cv2.THRESH_BINARY)
    # Here I add gaussian noise to the image, then I apply the 135 degree edge detection filter
    threshold, gauss_noise_filter_135_beach = cv2.threshold(inverse_diagonal_kernel(gaussian_noise(beach_gray)), 50, 255, cv2.THRESH_BINARY)

    # Here I add salt and pepper noise to the image, then apply the median denoising filter, after that I apply the 45 degree edge detection filter
    threshold, median_filter_45_beach = cv2.threshold(diagonal_kernel(cv2.medianBlur(salt_pepper_noise(beach_gray), 5)), 50, 255, cv2.THRESH_BINARY)
    # Here I add gaussian noise to the image, then apply the median denoising filter, after that I apply the 45 degree edge detection filter
    threshold, gauss_median_filter_45_beach = cv2.threshold(diagonal_kernel(cv2.medianBlur(gaussian_noise(beach_gray), 5)), 50, 255, cv2.THRESH_BINARY)
    
    # Here I add salt and pepper to the image, then apply the median denoising filter, after that I apply the 135 degree edge detection filter
    threshold, median_filter_135_beach = cv2.threshold(inverse_diagonal_kernel(cv2.medianBlur(salt_pepper_noise(beach_gray), 5)), 50, 255, cv2.THRESH_BINARY)
    # Here I add gaussian noise to the image, then apply the median denoising filter, after that I apply the 135 degree edge detection filter
    threshold, gauss_median_filter_135_beach = cv2.threshold(inverse_diagonal_kernel(cv2.medianBlur(gaussian_noise(beach_gray), 5)), 50, 255, cv2.THRESH_BINARY)

    # Binarize both filtered (kernal filters) images using a threshold value of 50
    threshold_value, binarized_45_beach = cv2.threshold(diagonal_kernel(beach_gray), 50, 255, cv2.THRESH_BINARY)
    threshold_value, binarized_135_beach = cv2.threshold(inverse_diagonal_kernel(beach_gray), 50, 255, cv2.THRESH_BINARY)

    spatial_edge_images = [binarized_45_beach, binarized_135_beach, noise_filter_45_beach, median_filter_45_beach, noise_filter_135_beach, median_filter_135_beach, gauss_noise_filter_45_beach, gauss_median_filter_45_beach, gauss_noise_filter_135_beach , gauss_median_filter_135_beach]
    spatial_edge_titles = ['45 degree edge filter', '135 degree edge filter', 'Noise with filter 45', 'Denoising with filter 45', 'Noise with filter 135', 'Denoising with filter 135', 'Gaussian noise filter 45', 'Denoising Gaussian', 'Gaussian noise filter 135', 'Denoising Gaussian']
    display_image(images=spatial_edge_images, rows=2, cols=5, figsize=(15,8), titles=spatial_edge_titles)

def salt_pepper_noise(image):
    # we get the height and width of the initial image and make a copy, so we have a blank image
    x, y = image.shape
    new_image = np.copy(image)

    # initialise salt and pepper values
    pepper = 0.001
    salt = 1 - pepper

    # Generate random values between 0 and 1
    rand = np.random.rand(x, y)

    # Here I replace the pixel with salt noise
    new_image[rand < pepper] = 255  
    # Here I replace the pixel with pepper noise
    new_image[rand > salt] = 0

    return new_image.astype(np.float32)

# This is the gaussian noise I am adding to my image
def gaussian_noise(image):
    # Here I create the gaussian noise with mean 0 and standard deviation 50
    # I add the noise to my add here and clamp the values of the pixel in the range (0,255)
    return np.clip(image + (np.random.normal(0, 50, image.shape)), 0, 255).astype(np.uint8)

# This is the distance computation used for the Butterworth filter
def distance_computation(height, width, u, v, sign):
    if sign == '+':
        return np.sqrt((u - height//2 - 0)**2 + ((v - width//2 - 99.9)**2))
    else:
        return np.sqrt((u - height//2 + 0)**2 + ((v - width//2 + 99.9)**2))

# This is the formula for the butterworth notch reject filter
def filter(d, n, height, width):
    h = np.ones((height, width))
    for u in range(height):
        for v in range(width):
             h[u,v] = (1 / (1 + (d/distance_computation(height, width, u, v, '+'))**n)) * (1 / (1 + (d/distance_computation(height, width, u, v, '-'))**n))
    return h

def fourier_transform():
    
    # convert the image to gray scale
    birdie_img_gray = cv2.cvtColor(birdie_img, cv2.COLOR_BGR2GRAY)
    birdie_img_gray = cv2.resize(birdie_img_gray, (720, 480))

    # I get the height and wide of my image here
    height, width = birdie_img_gray.shape
    
    # Here I initialise my chosen values for frequency and amplitude
    freq = 100
    amplitude = 50
    # I make a meshgrid of x,y coordinates
    x,y = np.meshgrid(np.arange(width), np.arange(height))
    # This is the cosine formula function for 2D and I display my image using my own function
    cosine = (amplitude * np.cos(2 * np.pi * freq * x / width) + amplitude).astype(np.uint8)

    # Here I find the 2D FFT
    fft_2d = np.fft.fft2(cosine)
    # I shift it to the center
    fft_2d_centered = np.fft.fftshift(fft_2d)

    # I calculate the magnitude of the FFT here 
    fft_2d_magnitude = (20 * np.log(np.abs(fft_2d_centered))).clip(0, 255).astype(np.uint8)
    
    
    
    # Here I shall extract the middle row and column of my magnitude spectrum
    middle_row = fft_2d_magnitude[fft_2d_magnitude.shape[0] // 2, :]
    middle_col = fft_2d_magnitude[:, fft_2d_magnitude.shape[1] // 2]
    
    # Adding period noise by using the 2D cosine noise and applying a FFT to it
    image_noisy = cv2.add(birdie_img_gray, cosine)
    fft_2d_cosine = np.fft.fft2(image_noisy)
    # Now I will center it in 2D
    fft_2d_cosine_centered = np.fft.fftshift(fft_2d_cosine)
    # Now I calculate the magnitude of my centered FFT here
    fft_2d_magnitude_noise = 20*np.log(np.abs(fft_2d_cosine_centered))
    display_image(images=[cosine, fft_2d_magnitude, fft_2d_magnitude_noise, image_noisy], figsize=(8,5), rows=2, cols= 2, titles=['2D cosine', 'FFT magnitude', 'FFT magnitude noise', 'Noisy Image'])

    # Here I shall extract the middle row and column of my magnitude spectrum
    middle_row_noise = fft_2d_magnitude_noise[fft_2d_magnitude_noise.shape[0] // 2, :]
    middle_col_noise = fft_2d_magnitude_noise[:, fft_2d_magnitude_noise.shape[1] // 2]
    display_image(images=[middle_row, middle_col, middle_row_noise, middle_col_noise], figsize=(12,8), rows=2, cols= 2, titles=['Middle row', 'Middle col', 'Middle row noise', 'Middle col noise'], mode='lineplot', axis='on')

    # Here I get the height and width of my image
    height, width = image_noisy.shape
    # Then I apply my butterwoth notch reject filter
    img_notch_filter = filter(10, 2, height, width)
    # I multiply it by frequencies already in the domain to get my magnitude for the filter
    fft_img = img_notch_filter * fft_2d_magnitude_noise
    # Now I do the inverse shifting to not center it in 2D
    inverse_shifted = np.fft.ifftshift(np.exp(fft_img/20 + 1j*np.angle(fft_2d_cosine_centered)))
    # Now I compute the inverse fourier transform 
    restored_img = np.abs(np.fft.ifft2(inverse_shifted)).clip(0, 255).astype(np.uint8)
    
    # Here I am display the magnitudes of the filter, magnitude of the denoised image and the denoised image
    display_image(images=[img_notch_filter, fft_img, restored_img], figsize=(12,8), rows=1, cols=3, titles=['Denoised filter magnitude', 'Denoised image magnitude', 'Denoised image'])
    # Here I display the 1D slices of my respective images
    display_image(images=[img_notch_filter[img_notch_filter.shape[0]//2, :], restored_img[restored_img.shape[0]//2, :]] , figsize=(12,8), rows=1, cols=2, titles=['Denoised Filter Slice', 'Denoised Image Slice'], mode='lineplot', axis='on')


def special_effects():
    # I apply gaussian blur here and convert it to HSV
    sheep_img_rgb = cv2.GaussianBlur(sheep_img, (31, 31), 0)  
    sheep_img_rgb = cv2.cvtColor(sheep_img_rgb, cv2.COLOR_BGR2HSV)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    _, label, center = cv2.kmeans(np.float32(sheep_img_rgb.reshape((-1,3))), 7, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert the centers to uint8 type
    center = np.uint8(center)
    # Convert the labels to 1D array    
    label = center[label.flatten()]

    # Here is the segmented image, where I am converting it back to rgb
    sheep_img_segmented = cv2.cvtColor(label.reshape(sheep_img_rgb.shape), cv2.COLOR_HSV2RGB)
    
    # I apply canny edge detection here and invert the edge, which is the mask I am applying on the image as the black outlines
    edges = cv2.Canny(sheep_img_segmented, 100, 200)
    inverted_edges = 255 - edges
    cartoon_img = cv2.bitwise_and(sheep_img_segmented, sheep_img_segmented, mask=inverted_edges)

    # Here I display the segmented image and the cartoon image
    display_image(images=[sheep_img_segmented, cartoon_img], figsize=(8,8), rows=1, cols=2, titles=['Segmented Image', 'Cartoon Image'])

def geometric_transform():
    # I get the height and width of my image
    height, width = palette_img.shape[:2]

    # I get the centers of the image here
    centerx, centery = width//2, height//2
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Here I convert the cartesian coordinate to polar coordinates
    rho = np.sqrt((x-centerx)**2 + (y-centery)**2)
    phi = np.arctan2(y-centery, x-centerx)

    # I am converting the polar coordinates back to cartesian form
    x_cart =  np.sqrt(rho) * np.cos((phi)) + centerx
    y_cart =  np.sqrt(rho) * np.sin((phi)) + centery

    # Finally, I get the transformed image by inversed mapping
    img_transformed = cv2.remap(palette_img, x_cart.astype(np.float32), y_cart.astype(np.float32), cv2.INTER_LINEAR)
    # Here I display the transformed image
    display_image(images=[cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)], figsize=(8,8), rows=1, cols=1, titles=['Image Transformed'])

def main():
    # color_spaces()
    # edge_detection_filter()
    fourier_transform()
    # special_effects() 
    # geometric_transform()


main()
