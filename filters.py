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

    gaussian_filter = np.zeros((m, n), np.float32)

    for x in range(-half_x, half_x):
        for y in range(-half_y, half_y):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
            gaussian_filter[x + half_x, y + half_y] = normal * exp_term

    blurred = convolution(img, gaussian_filter)

    return blurred.astype(np.uint8)


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('shape not supported')

    m_k, n_k = kernel.shape
    y_strides = m_i - m_k + 1
    x_strides = n_i - n_k + 1

    img = image.copy()
    output_shape = (m_i - m_k + 1, n_i - n_k + 1, c_i)
    output = np.zeros(output_shape, dtype=np.float32)

    count = 0
    output_tmp = output.reshape((output_shape[0] * output_shape[1], output_shape[2]))

    for y in range(y_strides):
        for x in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[y:y + m_k, x:x + n_k, c]
                output_tmp[count, c] = np.sum(sub_matrix * kernel)
            count += 1

    return output_tmp.reshape(output_shape)


def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)

