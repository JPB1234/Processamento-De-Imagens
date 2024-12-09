import cv2
from typing import Union, List, Tuple
import numpy as np
from image_handler import display_image

def thresholding_segmentation(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    binary_img = aplying_thresholding(gray_img)
    return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

def otsu_segmentation(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    binary_img = aplying_otsu(gray_img)
    return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

def erosion(img_cv, *args, kernel_size=5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    eroded_img = aplying_erosion(gray_img, kernel_size)
    return cv2.cvtColor(eroded_img, cv2.COLOR_GRAY2BGR)

def dilatation(img_cv, *args, kernel_size=5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    dilated_img = aplying_dilatation(gray_img, kernel_size)
    return cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)

def open(img_cv, *args, kernel_size=5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    segment_img = aplying_otsu(gray_img)
    eroded_img = aplying_erosion(segment_img, kernel_size)
    dilated_img = aplying_dilatation(eroded_img, kernel_size)
    return cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)

def close(img_cv, *args, kernel_size=5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    segment_img = aplying_thresholding(gray_img)
    dilated_img = aplying_dilatation(segment_img, kernel_size)
    eroded_img = aplying_erosion(dilated_img, kernel_size)
    return cv2.cvtColor(eroded_img, cv2.COLOR_GRAY2BGR)

def low_pass(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    return cv2.GaussianBlur(img_cv, (15, 15), 0)

def low_pass_gaussian(img_cv, *args, sigma=3.0, **kwargs):
    if img_cv is None:
        return None
    return GuassianBlur(img_cv, sigma)

def low_pass_media(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    return low_pass_mean_filter(img_cv)

def high_pass(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.Laplacian(gray_img, cv2.CV_64F)
    filtered_img = cv2.convertScaleAbs(filtered_img)
    return cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

def high_pass_laplacian(img_cv, kernel_value, *args, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel_size = kernel_value if kernel_value % 2 != 0 else kernel_value + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) * -1
    kernel[kernel_size // 2, kernel_size // 2] = kernel_size ** 2 - 1
    filtered_img = cv2.filter2D(gray_img, -1, kernel)
    return cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

def high_pass_sobel(img_cv, *args, direction='x', **kwargs):
    if img_cv is None:
        return None
    return sobel_filter_manual(img_cv, direction)

def GuassianBlur(img: np.ndarray, sigma: Union[float, int], filter_shape: Union[List, Tuple, None] = None):
    if filter_shape is None:
        shape = 2 * int(4 * sigma + 0.5) + 1
        filter_shape = [shape, shape]
    kernel = cv2.getGaussianKernel(filter_shape[0], sigma)
    gaussian_filter = np.outer(kernel, kernel)
    return cv2.filter2D(img, -1, gaussian_filter).astype(np.uint8)

def sobel_filter_manual(img, direction='x'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) if direction == 'x' else \
             np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_img = cv2.filter2D(gray, cv2.CV_64F, kernel)
    return cv2.convertScaleAbs(sobel_img)

def aplying_erosion(img_cv, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img_cv, kernel, iterations=1)

def aplying_dilatation(img_cv, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img_cv, kernel, iterations=1)

def aplying_otsu(gray_img):
    _, otsu_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_img

def aplying_thresholding(gray_img, threshold_value=90):
    _, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_img

def low_pass_mean_filter(img, kernel_size=9):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(img, -1, kernel)
