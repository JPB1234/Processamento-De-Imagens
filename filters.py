from typing import List, Tuple, Union
from matplotlib import pyplot as plt
import cv2
from image_handler import display_image
import numpy as np


def low_pass( img_cv, canvas):
    if img_cv is None:
        return
    filtered_img = cv2.GaussianBlur(img_cv, (15, 15), 0)
    display_image(filtered_img, canvas, original=False)
    
def low_pass_implemented(img_cv, canvas, sigma=1.0):
    if img_cv is None:
        return

    if not isinstance(img_cv, np.ndarray):
        img_cv = np.array(img_cv)

    filtered_img = GuassianBlur(img_cv, sigma)

    #Ocorreram alguns erros pq a imagem não era um np.array
    # TODO: Apagar condicional depoius de corrigir o erro.
    if not isinstance(filtered_img, np.ndarray):
        raise TypeError("not nparray")

    display_image(filtered_img, canvas, original=False)


def high_pass( img_cv, canvas):
    if img_cv is None:
        return
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.Laplacian(gray, cv2.CV_64F)
    filtered_img = cv2.convertScaleAbs(filtered_img)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    display_image(filtered_img, canvas, original=False)

def high_pass_implemented(img_cv, canvas):
    if img_cv is None:
        return
    
    if not isinstance(img_cv, np.ndarray):
        img_cv = np.array(img_cv)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])

    #kernel = np.array([[-1, -1, -1, -1, -1],
    #                       [-1,  1,  2,  1, -1],
    #                       [-1,  2,  4,  2, -1],
    #                       [-1,  1,  2,  1, -1],
    #                       [-1, -1, -1, -1, -1]])

    filtered_img = manual_convolution(gray, kernel)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    display_image(filtered_img, canvas, original=False)

def teste_high_pass_dinamic_kernel(img_cv, canvas, kernel_value):
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

    blurred = manual_convolution(img, gaussian_filter)

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