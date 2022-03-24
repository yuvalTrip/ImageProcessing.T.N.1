"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import cv2  # library of Python bindings designed to solve computer vision problems.
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 318916335


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    img_file_path = filename  # 'C:/Users/yuval/trainCatVSDogs/train/dog.10071.jpg' #r'C:\Users\yuval\queen.png'

    if representation == 1:  # If representation is grayscale
    # Using cv2.cvtColor() method
    # Using cv2.COLOR_BGR2GRAY color space
    # conversion code
        src = cv2.imread(img_file_path)  # Create the numpy array of the image
        img_array = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        normed_matrix = normalize(img_array) #normlize


    elif representation == 2:  # If representation is RGB
        img_array = cv2.imread(img_file_path,1)#Create the numpy array of the image.
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        img_a = img_array[:, :, 0]
        img_b = img_array[:, :, 1]
        img_c = img_array[:, :, 2]  # Extracting single channels from 3 channel image
        # The above code could also be replaced with cv2.split(img) << which will return 3 numpy arrays (using opencv)
        # normalizing per channel data(means we will normlize each diemention separately):
        img_a = (img_a - np.min(img_a)) / (np.max(img_a) - np.min(img_a))
        img_b = (img_b - np.min(img_b)) / (np.max(img_b) - np.min(img_b))
        img_c = (img_c - np.min(img_c)) / (np.max(img_c) - np.min(img_c))

        # putting the 3 channels back together:
        img_norm = np.empty(img_array.shape, dtype=np.float32) #create new empty np.array
        img_norm[:, :, 0] = img_a
        img_norm[:, :, 1] = img_b
        img_norm[:, :, 2] = img_c
        normed_matrix=img_norm


#By using below links:
#https://stackoverflow.com/questions/42460217/how-to-normalize-a-4d-numpy-array
#https://stackoverflow.com/questions/42460217/how-to-normalize-a-4d-numpy-array

    return normed_matrix


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img_array = imReadAndConvert(filename, representation)
    if representation == 1:  # If representation is grayscale
        plt.imshow(img_array, cmap="gray")
        plt.show()  # show the image

    elif representation == 2:  # If representation is RGB
        plt.imshow(img_array)
        plt.show()  # show the image


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    constant_mat = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]#define the constant mat with constant values
    Ans_mat = np.zeros_like(imgRGB.astype(float)) # create new np.array of zeroes

    shape = imgRGB.shape[0]
    for val in range(shape):#move over all pixels in image
        Ans_mat[val, ...] = np.matmul(imgRGB[val, ...], constant_mat)
    return Ans_mat

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    constant_mat = [[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]]#define the constant mat with constant values
    Ans_mat = np.zeros_like(imgYIQ.astype(float)) # create new np.array of zeroes

    shape = imgYIQ.shape[0]
    for val in range(shape):#move over all pixels in image
        Ans_mat[val, ...] = np.matmul(imgYIQ[val, ...], constant_mat)
    return Ans_mat


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
