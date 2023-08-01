import fnmatch
import math
import os
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

# This is the distance computation used for butterworth low/high pass filter
def distance_computation(height, width, u, v):
    return np.sqrt((u - height//2)**2 + (v - width//2)**2)

# This is the formula for butterworth low pass
def butterworth_low_pass(image, d0, n):
    h = np.zeros(image.shape[:2])
    height, width = image.shape[:2]
    for u in range(height):
        for v in range(width):
            h[u,v] = (1/(1 + (distance_computation(height, width, u, v)/d0)** (2*n)))
    return h

# This is the formula for butterworth high pass
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

    # Here I apply the low pass filter in the frequency domain and then inverse fourier transform to get the image
    img_low_pass = butterworth_low_pass(img_1, 20, 2)
    fft_img_1 = np.fft.fftshift(np.fft.fft2(img_1)) * img_low_pass
    img_1_restored = np.fft.ifft2(np.fft.ifftshift(fft_img_1)).clip(0, 255).astype(np.uint8)

    # Here I apply the high pass filter in the frequency domain and then inverse fourier transform to get the image
    img_high_pass = butterworth_high_pass(img_2, 20, 2)
    fft_img_2 = np.fft.fftshift(np.fft.fft2(img_2)) * img_high_pass
    img_2_restored = np.fft.ifft2(np.fft.ifftshift(fft_img_2)).clip(0, 255).astype(np.uint8)

    # I add the images in the spatial domain
    combined_img = cv2.add(img_1_restored, img_2_restored)
  
    # Here I display all respective images
    display_image(images=[img_1_restored, img_2_restored, combined_img], figsize=(10,5), rows=1, cols = 3, titles=['Low pass image', 'High pass image', 'Combined image'])

# Here is the function for performing 2D DCT
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.T, norm='ortho' ).T, norm='ortho' )

# Here is the function for performing inverse 2D DCT
def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.T , norm='ortho').T,norm='ortho')

# This is just a method for performing 8x8 2D DCT for viewing purposes, not really used
def blockwise_dct(img):
    imsize = img.shape
    dct = np.zeros(imsize)

    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            dct[i:i+8,j:j+8] = dct2( img[i:i+8,j:j+8])
    return dct

# This is a zigzag function to get the K-coefficients
def zigzag(input): 
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[0]
    hmax = input.shape[1]
    i = 0
    output = np.zeros((vmax * hmax))
    
    while ((v < vmax) and (h < hmax)):

        if ((h + v) % 2) == 0:
            if (v == vmin):
                output[i] = input[v,h]
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            else:
                if ((h == hmax-1) and (v < vmax)):
                    output[i] = input[v,h]
                    v = v + 1
                    i = i + 1
                else:
                    if ((v > vmin) and (h < hmax-1)):
                        output[i] = input[v,h]
                        v = v - 1
                        h = h + 1
                        i = i + 1
        else:
            if ((v == vmax-1) and (h <= hmax-1)):
                output[i] = input[v,h]
                h = h + 1
                i = i + 1
            else:
                if (h == hmin):
                    output[i] = input[v,h]
                    if (v == vmax-1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                else:
                    if ((v < vmax-1) and (h > hmin)):
                        output[i] = input[v,h]
                        v = v + 1
                        h = h - 1
                        i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):
            output[i] = input[v,h]
            break

    
    return output

# Here is the inverse zig-zag function to get it back to the 8x8 matrix
def inverse_zigzag(input, vmax, hmax):
	
	#print input.shape

	# initializing the variables
	#----------------------------------
	h = 0
	v = 0

	vmin = 0
	hmin = 0

	output = np.zeros((vmax, hmax))

	i = 0
    #----------------------------------

	while ((v < vmax) and (h < hmax)): 
		#print ('v:',v,', h:',h,', i:',i)   	
		if ((h + v) % 2) == 0:                 # going up
            
			if (v == vmin):
				#print(1)
				
				output[v, h] = input[i]        # if we got to the first line

				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[v, h] = input[i] 
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[v, h] = input[i] 
				v = v - 1
				h = h + 1
				i = i + 1

        
		else:                                    # going down

			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[v, h] = input[i] 
				h = h + 1
				i = i + 1
        
			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[v, h] = input[i] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1
        		        		
			elif((v < vmax -1) and (h > hmin)):     # all other cases
				output[v, h] = input[i] 
				v = v + 1
				h = h - 1
				i = i + 1




		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)        	
			output[v, h] = input[i] 
			break


	return output

# This is the method for embedding the watermark in the image
def embedding_watermark(watermark, image, alpha, img_restored, K):
    # Here for each 8x8 block in the image, I get the coefficients and modify the K-most important coefficients with the watermark value
    for i in np.r_[:image.shape[0]:8]:
        for j in np.r_[:image.shape[1]:8]:
            # for each block I perform 2D DCT and get the K-coefficients through zigzag scan
            block = image[i:(i+8),j:(j+8)]
            img_dct = dct2(block)
            coefficients = zigzag(img_dct)

            # Embedding watermark values into K-most important coefficients
            coefficients[1:K+1] *= (1+alpha*watermark)

            # Create a 8x8 matrix to store back the matrix after inverse zigzag
            inv_zigzag = np.zeros((8,8)) 
            inv_zigzag = inverse_zigzag(coefficients, 8, 8)

            # Performing inverse 2D DCT to get back my image after embedding watermark
            inv_dct = idct2(inv_zigzag)
            img_restored[i:(i+8),j:(j+8)] = inv_dct

    return img_restored

# This method is for detecting a watermark in an image
def detect_watermark(watermark, image, alpha, K):
     correlation_coeff = 0
     for i in np.r_[:image.shape[0]:8]:
        for j in np.r_[:image.shape[1]:8]:
            # for each block I perform 2D DCT and get the K-coefficients through zigzag scan
            block = image[i:(i+8),j:(j+8)]
            img_dct = dct2(block)
            coefficients = zigzag(img_dct)
            # Here I calculate the new watermark value to measure the correlation coefficient between old watermark and new watermark
            coefficients_new = zigzag(img_dct)
            coefficients_new[1:K+1] *= (1+alpha*watermark)
            w = (coefficients_new[1:K+1] - coefficients[1:K+1]) / (alpha*coefficients[1:K+1])
            correlation_coeff = np.sum((w - np.mean(w)) * (watermark - np.mean(watermark))) / np.sqrt(np.sum((w - np.mean(w))**2) * np.sum((watermark - np.mean(watermark))**2))
     return correlation_coeff

def watermark():
    # Here I read in the images I need
    kirby_img = cv2.imread('images-project2/kirby.jpg')
    bike = cv2.imread('images-project2/bike.jpg')
    kirby_gray = cv2.cvtColor(kirby_img, cv2.COLOR_BGR2GRAY)
    bike_gray = cv2.cvtColor(bike, cv2.COLOR_BGR2GRAY)

    # Here I make the respective matrix sizes which I will need later for the iamges restored and for display
    img_dct = np.zeros((kirby_gray.shape))
    img_restored = np.zeros((kirby_gray.shape))
    img_watermark = np.zeros((kirby_gray.shape))
    img_mystery_1 = np.zeros((kirby_gray.shape))
    img_mystery_2 = np.zeros((kirby_gray.shape))

    # K is the value you modify for the K-coefficients from the zigzag scan
    K = 1
    coefficients = 0
    # 25 watermark values generated pulled from a gaussian distribution
    np.random.seed(10)
    watermark = np.random.normal(0, 1, 25)
    watermark_new = np.random.normal(0, 5, 25)

    for i in np.r_[:kirby_gray.shape[0]:8]:
        for j in np.r_[:kirby_gray.shape[1]:8]:
            block = kirby_gray[i:(i+8),j:(j+8)]
            # for each block I perform 2D DCT and get the K-coefficients through zigzag scan
            img_dct = dct2(block)
            coefficients = zigzag(img_dct)
            coefficients[K+1:] = 0

            # Create a 8x8 matrix to store back the matrix after inverse zigzag
            inv_zigzag = np.zeros((8,8)) 
            inv_zigzag = inverse_zigzag(coefficients, 8, 8)

            # Performing inverse 2D DCT to get back my image 
            inv_dct = idct2(inv_zigzag)
            img_restored[i:(i+8),j:(j+8)] = inv_dct

    # Calling method to get image from embedded watermark
    watermark_img = embedding_watermark(watermark, kirby_gray, 0.2, img_watermark, 25)
    # I get the difference image between original and watermark image
    difference_img = kirby_gray - watermark_img

    # Here I create two new mystery images with embedded watermarks
    mysteryimg_1 = embedding_watermark(watermark, bike_gray , 0.2, img_mystery_1, 25)
    mysteryimg_2 = embedding_watermark(watermark_new, bike_gray, 0.2, img_mystery_2, 25)
    

    # Here you can view the values for the correlation coefficients, when you print them
    gamma_1 = detect_watermark(watermark, mysteryimg_1, 0.2, 25)
    gamma_2 = detect_watermark(watermark_new, mysteryimg_2, 0.2, 25)
    print("gamma 1: ", gamma_1)
    print("gamma 2: ", gamma_2)
    # Here I make the following decision of when an image has a watermark
    # Essentially gamma > 0 implies a watermark being present
    # Otherwise there is none present 
    if(gamma_1 >= 0.5):
         print("Gamma 1 has a watermark present")
    else:
         print("There is no watermark present")
    
    if(gamma_2 >= 0.5):
         print("Gamma 2 has a watermark present")
    else:
         print("There is no watermark present")

    # Here I display all the final images
    display_image(images=[kirby_gray, img_restored, watermark_img, difference_img, mysteryimg_1, mysteryimg_2], figsize=(10,5), rows=2, cols = 3, 
                     titles=['Original image', 'K coefficients', 'Watermark image', 'Difference Image', 'Mystery 1', 'Mystery 2'])
    display_image(images=[np.abs(difference_img)], figsize=(10,5), rows=1, cols=1, titles=['Difference Image Hist'], mode='histogram', axis='on')



def MEI_for_image():

    # This commented code below gets the frames I need which goes into the folder frames/binary     
    
    #  walking = cv2.VideoCapture('walking.avi')
    #  # reading frames to get the initial frames
    #  ret1, frame1 = walking.read()
    #  # We make it grayscale
    #  frame_prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #  hsv = np.zeros_like(frame1)
    #  hsv[...,1] = 255
    #  frame_counter = 59
    #  output_folder = "frames_binary"
    #  os.makedirs(output_folder, exist_ok=True)
    #  # I loop over 50 times and increment my frame counter 10 times, therefore producing 500 frames, saved into a folder 
    #  # where I can view the frames of my image
    #  for i in range(50):
    #     # Here we read to get the consecutive frames
    #     ret2, frame2 = walking.read()
    #     # We make it grayscale
    #     frame_next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    #     # In-built function to calculate the optical flow
    #     flow  = cv2.calcOpticalFlowFarneback(frame_prev, frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     # convert cartesian to polar coordinates
    #     magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #     # We get the h,v components for the hsv here
    #     hsv[..., 0] = angle*(180/(np.pi/2))
    #     hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    #     # We convert it back to to the BGR space for display
    #     hsv_to_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    #     # Here I am incrementing my frame counter and creating the folder, where my output images will go to
    #     frame_counter += 1 
    #     output = os.path.join(output_folder, f"frame_{frame_counter}.png")
    #     cv2.imwrite(output, hsv_to_rgb)

    #     # Setting the next frame to the previous frame
    #     frame_prev = frame_next
     
     # After getting my images into my folder, I read the folder of images

     for file in os.listdir('frames_binary/'):
            img_path = os.path.join('frames_binary/', file)
            # Convert it to grayscale
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Here I have a threshold of 40 and binarize my image
            _, binary = cv2.threshold(frame, 40, 255, cv2.THRESH_BINARY)
            resized_img = cv2.resize(binary, (500, 500))

            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(resized_img, cv2.MORPH_OPEN, kernel)

            # I apply canny edge detection to get the outlines
            edges = cv2.Canny(morph, 100, 200)

            # Here I extract the shape descriptors for the HuMoments
            moment = cv2.moments(edges)
            huMoment = cv2.HuMoments(moment)
            print(huMoment)

            # Below you can see the edges and binary images by uncommenting one of the imshow and commenting the other
            # Basically leave the imshow uncommented based on which images you want to sees
            # cv2.imshow('Binary', resized_img)
            # cv2.imshow('Morph', morph)
            cv2.imshow('Outline', edges)
            cv2.waitKey(0)
     cv2.destroyAllWindows()  
     
     


# This method if for getting the motion energy images
def MEI_for_video():
     # I read the following videos I will be using
     walking = cv2.VideoCapture('walking.avi')
     handclapping = cv2.VideoCapture('handclapping.avi')

    # reading frames to get the initial frames
     ret1, frame1 = handclapping.read()
     # We make it grayscale
     frame_prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
     hsv = np.zeros_like(frame1)
     hsv[...,1] = 255
     frame_counter = 0
     output_folder = "frames_handclapping"
     os.makedirs(output_folder, exist_ok=True)
     # I loop over 50 times and increment my frame counter 10 times, therefore producing 500 frames, saved into a folder 
     # where I can view the frames of my image
     for i in range(50):
        # Here we read to get the consecutive frames
        ret2, frame2 = handclapping.read()
        # We make it grayscale
        frame_next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # In-built function to calculate the optical flow
        flow  = cv2.calcOpticalFlowFarneback(frame_prev, frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # convert cartesian to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # We get the h,v components for the hsv here
        hsv[..., 0] = angle*(180/(np.pi/2))
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # We convert it back to to the BGR space for display
        hsv_to_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Here I am incrementing my frame counter and creating the folder, where my output images will go to
        frame_counter += 10 
        output = os.path.join(output_folder, f"frame_{frame_counter}.png")
        cv2.imwrite(output, hsv_to_rgb)

        # Setting the next frame to the previous frame
        frame_prev = frame_next
        
     
     walking.release()
     cv2.destroyAllWindows()
def main():
    human_perception()  
    watermark()
    # MEI_for_video()
    # MEI_for_image()

main()