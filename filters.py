import cv2
from typing import Union, List, Tuple
import numpy as np
from image_handler import display_image

def erosion(img_cv, canvas, kernel_size = 5):
    if img_cv is None:
        return
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    output_img = aplying_erosion(img_cv,kernel_size)
    eroded_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    display_image(eroded_img, canvas, original=False)


def dilatation(img_cv, canvas, kernel_size=5):
    if img_cv is None:
        return
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    output_img = aplying_dilatation(img_cv,kernel_size)
    
    dilated_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    display_image(dilated_img, canvas, original=False)

def open(img_cv, canvas, kernel_size=5):
    if img_cv is None:
        return
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    erosion_img = aplying_dilatation(img_cv,kernel_size)
    dilated_img = aplying_dilatation(erosion_img,kernel_size)
    
    final_img = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
    display_image(final_img, canvas, original=False) 

def close(img_cv, canvas, kernel_size=5):
    if img_cv is None:
        return
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    dilated_img = aplying_dilatation(img_cv,kernel_size)
    erosion_img = aplying_dilatation(dilated_img,kernel_size)
    
    final_img = cv2.cvtColor(erosion_img, cv2.COLOR_GRAY2BGR)
    display_image(final_img, canvas, original=False) 

    
def low_pass(img_cv, canvas):
    if img_cv is None:
        return
    filtered_img = cv2.GaussianBlur(img_cv, (15, 15), 0)
    display_image(filtered_img, canvas, original=False)

def low_pass_gaussian(img_cv, canvas, sigma=3.0):
    if img_cv is None:
        return

    if not isinstance(img_cv, np.ndarray):
        img_cv = np.array(img_cv)

    filtered_img = GuassianBlur(img_cv, sigma)
    if not isinstance(filtered_img, np.ndarray):
        raise TypeError("not nparray")
    
    display_image(filtered_img, canvas, original=False)

def low_pass_media(img_cv, canvas):
    if img_cv is None:
        return

    if not isinstance(img_cv, np.ndarray):
        img_cv = np.array(img_cv)

    filtered_img = low_pass_mean_filter(img_cv)
    if not isinstance(filtered_img, np.ndarray):
        raise TypeError("not nparray")
    
    display_image(filtered_img, canvas, original=False)

def high_pass(img_cv, canvas):
    if img_cv is None:
        return
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.Laplacian(gray, cv2.CV_64F)
    filtered_img = cv2.convertScaleAbs(filtered_img)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    display_image(filtered_img, canvas, original=False)

def high_pass_laplacian(img_cv, canvas, kernel_value):
    if img_cv is None:
        return

    if not isinstance(img_cv, np.ndarray):
        img_cv = np.array(img_cv)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    kernel_size = kernel_value
    if kernel_size % 2 == 0:
        kernel_size += 1 

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) * -1
    kernel[kernel_size // 2, kernel_size // 2] = kernel_size ** 2 - 1

    filtered_img = cv2.filter2D(gray, -1, kernel)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    display_image(filtered_img, canvas, original=False)

def high_pass_sobel(img_cv, canvas):
    if img_cv is None:
        return

    if not isinstance(img_cv, np.ndarray):
        img_cv = np.array(img_cv)
    
    filtered_img = sobel_filter_manual(img_cv)
    display_image(filtered_img, canvas, original=False)

def GuassianBlur(img: np.ndarray, sigma: Union[float, int], filter_shape: Union[List, Tuple, None] = None):
    if filter_shape is None:
        shape = 2 * int(4 * sigma + 0.5) + 1
        filter_shape = [shape, shape]
    elif len(filter_shape) != 2:
        raise Exception('shape not supported')

    x, y = filter_shape
    half_x = x // 2
    half_y = y // 2

    gaussian_filter = np.zeros((x, y), np.float32)

    for x in range(-half_x, half_x):
        for y in range(-half_y, half_y):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
            gaussian_filter[x + half_x, y + half_y] = normal * exp_term

    blurred = cv2.filter2D(img, -1, gaussian_filter)
    return blurred.astype(np.uint8)

def manual_convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros_like(image)
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    for i in range(pad_h, image_height - pad_h):
        for j in range(pad_w, image_width - pad_w):
            sub_matrix = image[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            output[i, j] = np.sum(sub_matrix * kernel)
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

def low_pass_mean_filter(img, kernel_size=9):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img

def sobel_filter_manual(img, direction='x'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if direction == 'x':
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # Sobel X
    elif direction == 'y':
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  # Sobel Y
    sobel_img = cv2.filter2D(gray, cv2.CV_64F, kernel)

    return cv2.convertScaleAbs(sobel_img)

def aplying_erosion(img_cv, kernel_size = 5):
    pad_size = kernel_size // 2
    padded_img = np.pad(img_cv, pad_size, mode='constant', constant_values=255)
    output_img = np.zeros_like(img_cv)

    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            region = padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            output_img[i - pad_size, j - pad_size] = np.min(region)
    return output_img

def aplying_dilatation(img_cv, kernel_size = 5):
    pad_size = kernel_size // 2
    padded_img = np.pad(img_cv, pad_size, mode='constant', constant_values=0)
    
    output_img = np.zeros_like(img_cv)

    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            region = padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            output_img[i - pad_size, j - pad_size] = np.max(region)
    return output_img