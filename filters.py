from typing import List, Tuple, Union
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

def high_pass_implemented(img_cv, canvas, sigma=1.0):
    if img_cv is None:
        return
    
    blurred_img = GuassianBlur(img_cv, sigma)
    
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

    laplacian_filtered = cv2.Laplacian(gray_img, cv2.CV_64F)
    
    sobelx_filtered = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely_filtered = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)

    sobel_combined = cv2.magnitude(sobelx_filtered, sobely_filtered)

    filtered_img = cv2.convertScaleAbs(sobel_combined + laplacian_filtered)
    display_image(filtered_img, canvas, original=False)


def GuassianBlur(img: np.ndarray, sigma: Union[float, int], filter_shape: Union[List, Tuple, None] = None):
    if filter_shape is None:
        _ = 2 * int(4 * sigma + 0.5) + 1
        filter_shape = [_, _]
    elif len(filter_shape) != 2:
        raise Exception('shape of argument `filter_shape` is not supported')

    m, n = filter_shape
    m_half = m // 2
    n_half = n // 2

    gaussian_filter = np.zeros((m, n), np.float32)

    for y in range(-m_half, m_half):
        for x in range(-n_half, n_half):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
            gaussian_filter[y + m_half, x + n_half] = normal * exp_term

    blurred = convolution(img, gaussian_filter)

    return blurred.astype(np.uint8)


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('Shape of image not supported')

    m_k, n_k = kernel.shape
    y_strides = m_i - m_k + 1
    x_strides = n_i - n_k + 1

    img = image.copy()
    output_shape = (m_i - m_k + 1, n_i - n_k + 1, c_i)
    output = np.zeros(output_shape, dtype=np.float32)

    count = 0
    output_tmp = output.reshape((output_shape[0] * output_shape[1], output_shape[2]))

    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[i:i + m_k, j:j + n_k, c]
                output_tmp[count, c] = np.sum(sub_matrix * kernel)
            count += 1

    return output_tmp.reshape(output_shape)

