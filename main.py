from typing import List
import numpy as np
import cv2  # library of Python bindings designed to solve computer vision problems.
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
    src = cv2.imread(img_file_path)  # Create the numpy array of the image

    if representation == 1:  # If representation is grayscale
    # Using cv2.cvtColor() method
    # Using cv2.COLOR_BGR2GRAY color space
    # conversion code
        image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    elif representation == 2:  # If representation is RGB
        img_array = image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    normed_matrix = normalize(img_array, axis=1, norm='l1')

    return normed_matrix




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(imReadAndConvert('C:/Users/yuval/trainCatVSDogs/train/dog.10071.jpg',1))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
