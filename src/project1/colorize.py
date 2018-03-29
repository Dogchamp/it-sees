# Computer Vision Project 1 - Glass Plate Color Channel Alignment
# Uses an exhaustive Gaussian Pyramid coarse-to-fine search to align

import scipy
from scipy import signal
import numpy as np
import cv2

# ======================================  Gaussian Pyramid   ======================================
# Returns a Gaussian kernel meant for use as a Gaussian low pass filter
def make_g_kernel(a):
	kernel = np.array([0.25-(a/2.0), 0.25, a, 0.25, 0.25-(a/2.0)])
	return np.outer(kernel, kernel)

# Passes image through a gaussian filter and then reduces it by half
def scaledown(image):
	g_kernel = make_g_kernel(0.4)
	convolved = signal.convolve2d(image, g_kernel, 'same')
	reduced_im = convolved[::2, ::2]
	return reduced_im

# Returns gaussian pyramid of image 'image' levels 'levels' deep
def g_pyr(image, levels):
	out = []
	out.append(image)
	tmp = image
	for level in range(0, levels):
		tmp = scaledown(tmp)
		out.append(tmp)
	return out
# ==================================== ALIGNMENT ========================================
def gradient(image):
    x_gradient = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)
    y_gradient = cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)
    gradient_sum = cv2.addWeighted(np.absolute(x_gradient), 0.5, np.absolute(y_gradient), 0.5, 0)
    return gradient_sum

def get_alignment(im_a, im_b):
    """
    Compares Soble gradients to get a warp matrix to be used in a matrix affine transformation
    """
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)     # Initial
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    return cv2.findTransformECC(im_a, im_b, warp_matrix, warp_mode, criteria)

def align_by_gradient(im_a, im_b):
    """
    Gets warp matrix via gradient comparison and applies it to image_a.
    Returns new aligned image
    """
    cc, warp_matrix = get_alignment(gradient(im_a), gradient(im_b))
    return cv2.warpAffine(im_a, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# ============================= Image IO ==============================
def show_image(im, desc):
    cv2.imshow(desc, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======================================  MAIN  ======================================
# name of the input image
path = "./assets/"
imname = path + 'cathedral'
fname = imname + '.jpg'

# Read in the image as grayscale
im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

# get the height and width of each image
total_sz = im.shape
height = int(total_sz[0]/3)
width = total_sz[1]

# Bells and Whistles - Autocontrast using CLAHE
# (Contrast Limited Adaptive Histograme Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clim = clahe.apply(im)
print( "Applying autocontrast..." )
show_image(im, 'nocontrast')
show_image(clim, 'contrast')
cv2.imwrite(imname + '_clahe.jpg', clim)

# Separate picture into color channels from top to bottom
r = im[:height]
g = im[height:2*height]
b = im[2*height:3*height]

stacked = np.dstack((r,g,b))
show_image(stacked, "stacked")

aligned_image = np.zeros((height/3, width, 3), dtype=np.uint8)


# Align em
rg = align_by_gradient(r, g)
rgb = align_by_gradient(rg, b)

show_image(rgb, "rgb")
"""
