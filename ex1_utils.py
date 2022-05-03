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
import math  # to import math module

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
    if representation != 1 and representation != 2:
        raise ValueError(
            'only 1 or 2 are possible inputs for representation, ''to output Grayscale or RGB pics respectively')
    filepath = filename
    if representation == 1:  # if grayScale
        channel = 0
    elif representation == 2:  # if RGB
        channel = 1
    img = cv2.imread(filepath, channel)
    if img is None:  # raised an error
        raise ValueError("Could not find requested file inside folder. ")
    if channel:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # normlized values to 0-1
    return img


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

    constant_mat = [[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]]  # define the constant mat with constant values
    Ans_mat = np.zeros_like(imgRGB.astype(float))  # create new np.array of zeroes

    shape = imgRGB.shape[0]
    for val in range(shape):  # move over all pixels in image
        Ans_mat[val, ...] = np.matmul(imgRGB[val, ...], constant_mat)
    return Ans_mat


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    constant_mat = [[1, 0.956, 0.619],
                    [1, -0.272, -0.647],
                    [1, -1.106, 1.703]]  # define the constant mat with constant values
    Ans_mat = np.zeros_like(imgYIQ.astype(float))  # create new np.array of zeroes

    shape = imgYIQ.shape[0]
    for val in range(shape):  # move over all pixels in image
        Ans_mat[val, ...] = np.matmul(imgYIQ[val, ...], constant_mat)
    return Ans_mat


def Regular_histogram_Building(imgOrig: np.ndarray) -> (np.ndarray, int):
    """
    :param imgOrig: the array of an image
    :return: The array of a regular histogram of an image,
            and number represent the image dimension - 1 if grayscale and 2 if RGB
    """
    if imgOrig.ndim == 2:  # if this is gray scale
        # plt.imshow(imgOrig)
        # plt.show()
        # first , we will count the unique values of the array
        # because we get normalized values we will multiply each element in 255 and round it
        unique, counts = np.unique((np.round(imgOrig * 255)), return_counts=True)
        Ans_mat = np.zeros(256)  # create new np.array of zeroes in size of 256 because of the values [0,255]

        for i in range(len(counts)):
            Ans_mat[int(unique[i])] = counts[i]
            # example:
            # counts:    5 10 1 20 1
            # unique:    0 5 20 30 255
            # ans_mat:   5 0 0 0 0 10 0 0 ...1
            # i = i + 1
        finalArr_beforeCumSum = Ans_mat
        return finalArr_beforeCumSum, 1  # return the final array of regular histogram (1 if grayscale)

    elif imgOrig.ndim == 3:  # if this is RGB
        # first, as written in PDF, we will convert the RGB to YIQ image
        YIQimage = transformRGB2YIQ(imgOrig)  # convert the RGB array to YIQ
        Ychannel = YIQimage[:, :,
                   0]  # we will extract the first column as Y channel (those are normalized values between 0-1)
        # we want to turn the normalized values to values between 0-255,
        Ychannel = cv2.normalize(Ychannel, None, 0, 255, cv2.NORM_MINMAX)
        # because those are float numbers, we will round them
        Ychannel = np.ceil(Ychannel)  # floating point precision correction
        Ychannel = Ychannel.astype('uint8')  # turn all number to int
        #  count the unique values of the array
        (unique, counts) = np.unique(np.round(Ychannel), return_counts=True)

        newY = np.zeros(256)  # create new np.array of zeroes in size of 256 because of the values [0,255]
        i = 0
        for value in counts:
            newY[int(unique[i])] = value
            # example:
            # counts:    5 10 1 20 1
            # unique:    0 5 20 30 255
            # ans_mat:   5 0 0 0 0 10 0 0 ...1
            i = i + 1
        finalArr_beforeCumSum = newY
        return finalArr_beforeCumSum, 2  # return the final array of regular histogram (2 if RGB)


def Cumsum_histogram_Building(histArray: np.ndarray) -> (np.ndarray):
    """
    :param histArray: the array we got from the Regular_histogram_Building function
    :return: array of the cumsum
    """
    sum_mat = np.zeros(256)  # create new np.array of zeroes in size of 256 because of the values [0,255]
    i = 0
    for value in histArray:
        if i == 0:
            sum_mat[i] = value  # first element will be the same
        else:
            sum_mat[i] = sum_mat[i - 1] + value  # sum all numbers

        i = i + 1  # increase the index

    return sum_mat


def LUT_Building(SumArr: np.ndarray) -> (np.ndarray):
    """

    :param SumArr: The CumSum array of the image
    :return: Look Up Table as array
    """
    LUTarr = np.zeros(256)  # create new np.array of zeroes in size of 256 because of the values [0,255]
    for i in range(len(SumArr)):
        LUTarr[i] = math.ceil(
            (SumArr[i] * 255) / (SumArr[len(SumArr) - 1]))  # exactly the same formula as written in the PDF
    return LUTarr


def ApplyLUT(lutArr: np.ndarray, imOrig: np.ndarray) -> (np.ndarray):
    """
    :param lutArr: The lut array we created in LUT_Building
           imOrig: The array of the original image
    :return: the original array after we apply the LUT on it
    """
    tempimOrig = np.copy(imOrig)
    if imOrig.ndim == 2:  # if this is gray scale
        for i in range(len(tempimOrig)):  # move over all rows
            for k in range(len(tempimOrig[0])):  # move over all columns
                tempimOrig[i][k] = lutArr[int(np.round(imOrig[i][k] * 255))]
        return tempimOrig


    elif imOrig.ndim == 3:  # if this is RGB
        imgEq = transformRGB2YIQ(imOrig)  # we will transform it to YIQ (values between 0-1)
        imgEq[..., 0] = (imgEq[..., 0] - np.min(imgEq[..., 0])) / (
                    np.max(imgEq[..., 0]) - np.min(imgEq[..., 0])) * 255  # normlized Y channel to values 0-255
        for i in range(len(imgEq)):  # move over all rows
            for k in range(len(imgEq[0])):  # move over all columns
                newVal = lutArr[int(np.round(imgEq[i][k][0]))]
                imgEq[i][k][0] = newVal  # we will put the value we want from LUT in the new array, in the right index
        # after applying the LUT , we get values in Ychannel between 0-255
        imgEq[..., 0] = imgEq[..., 0] / 255  # we will normalized y channel to values between 0-1
        imgEq = transformYIQ2RGB(imgEq)  # we will return it to RGB image
        return imgEq


def hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
            Equalizes the histogram of an image
            :param imgOrig: Original Histogram
            :ret
        """
    histOrig, ColorModel = Regular_histogram_Building(imOrig)  # Original Histogram
    cumSum = Cumsum_histogram_Building(histOrig)  # CumSum
    # plt.title('Original image histogram with CDF') #display them
    # plt.plot(cumSum, color='b')
    # plt.hist(histOrig.flatten(), 256, [0, 255], color='r')
    # plt.xlim([0, 255])
    # plt.legend(('cdf - ORIGINAL', 'histogram - ORIGINAL'), loc='upper left')
    lutArr = LUT_Building(cumSum)
    imgAfterLUT = ApplyLUT(lutArr, imOrig)  # image equlized

    imgAfterLUT = (imgAfterLUT - float(np.min(imgAfterLUT)) / (float(np.max(imgAfterLUT) - np.min(imgAfterLUT))))

    histEq, ColorModel = Regular_histogram_Building(imgAfterLUT)  # Original Histogram

    # display the original image
    cv2.imshow('ORIGINAL image', imOrig)

    # display equalized image
    cv2.imshow('EQUALIZED image', imgAfterLUT)

    return imgAfterLUT, histOrig, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if (len(imOrig.shape) != 3 and len(imOrig.shape) != 2):
        raise ValueError('Unsupported array representation. only RGB or Grayscale images allowed')
    if len(imOrig.shape) == 3:  # If this is RGB (3D)
        imYIQ = transformRGB2YIQ(imOrig)
        imY = imYIQ[:, :, 0].copy()  # take only y channel
    else:  # If this is GrayScale (2D)
        imY = imOrig
    histOrig = np.histogram(imY.flatten(), bins=256)[0]  # Original Histogram
    Z, Q = Find_BestCenters(histOrig, nQuant, nIter)
    imHistory = [imOrig.copy()]
    E = []
    for i in range(len(Z)):
        arrayQuant = np.array([Q[i][k] for k in range(len(Q[i])) for x in range(Z[i][k], Z[i][k + 1])])
        q_img, e = convertToImg(imY, histOrig, imYIQ if len(imOrig.shape) == 3 else [], arrayQuant)
        imHistory.append(q_img)
        E.append(e)
    return imHistory, E


def Find_BestCenters(histOrig: np.ndarray, nQuant: int, nIter: int) -> (np.ndarray, np.ndarray):
    """
            function find the best nQuant centers for quantize the image in nIter steps given us, *or* when the error is minimum
            :param histOrig: hist of the original image
            :param nQuant: Number of colors to quantize the image to
            :param nIter: Number of optimization loops
            :return: return all centers and their color selected to build from it all the images.
        """
    Q = []
    Z = []
    # head start, all the intervals are in the same length
    z = np.arange(0, 256, round(256 / nQuant))
    z = np.append(z, [255])
    Z.append(z.copy())
    q = fix_q(z, histOrig)
    Q.append(q.copy())
    for n in range(nIter):
        z = fix_z(q)
        if (Z[-1] == z).all():  # if nothing changed we will break
            break
        Z.append(z.copy())
        q = fix_q(z, histOrig)
        Q.append(q.copy())
    return Z, Q


def fix_z(q: np.array) -> np.array:
    """
        Calculate new z, using the formula from the lecture:
        z[i]=q[i-1]-q[i]\2
        :param q: the new list of q
        :return: the new z- after fixing the boundary
    """
    z_new = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)  # we will find the middle
    z_new = np.concatenate(([0], z_new, [255]))
    return z_new


def fix_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
    """
        Calculate new q using wighted average on the histogram
        :param image_hist: the histogram of the original image
        :param z: the new list of centers
        :return: the new list of wighted average
    """
    q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=image_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
    return np.round(q).astype(int)


def convertToImg(imOrig: np.ndarray, histOrig: np.ndarray, imYIQ: np.ndarray, arrayQuantize: np.ndarray) -> (
np.ndarray, float):
    """
        Executing the quantization to the original image
        :return: returning the resulting image and the MSE.
    """
    imageQ = np.interp(imOrig, np.linspace(0, 1, 255), arrayQuantize)
    curr_hist = np.histogram(imageQ, bins=256)[0]
    err = np.sqrt(np.sum((histOrig.astype('float') - curr_hist.astype('float')) ** 2)) / float(
        imOrig.shape[0] * imOrig.shape[1])
    if len(imYIQ):  # if the original image is RGB
        imYIQ[:, :, 0] = imageQ / 255
        return transformYIQ2RGB(imYIQ), err
    return imageQ, err
