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
from sklearn.preprocessing import normalize, MinMaxScaler
from skimage import io
from sklearn.cluster import KMeans
#from sklearn.utils import
import math   #to import math module

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

#     img_file_path = filename
#
#     if representation == 1:  # If representation is grayscale
#     # Using cv2.cvtColor() method
#     # Using cv2.COLOR_BGR2GRAY color space
#     # conversion code
#         src = cv2.imread(img_file_path)  # Create the numpy array of the image
#         img_array = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#         normed_matrix = normlized(img_array,1) #normlize
#         return normed_matrix
#
#     elif representation == 2:  # If representation is RGB
#         img_array = cv2.imread(img_file_path)#Create the numpy array of the image.
#         img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#         normed_matrix = normlized(img_array,2) #normlize
#         return normed_matrix
#
# def normlized (img:np.ndarray ,representation: int)->np.ndarray:
#     """
#
#     :param img:
#     :param representation:
#     :return:
#     """
#     min_val=img.min()
#     max_val=img.max()
#     normlized_arr=(img- float(min_val)) /float(max_val-min_val)
#     return normlized_arr
    if representation != 1 and representation != 2:
            raise ValueError('only 1 or 2 are possible inputs for representation, ''to output Grayscale or RGB pics respectively')
    filepath = filename
    if representation == 1:
        channel = 0
    elif representation == 2:
        channel = 1
    img = cv2.imread(filepath, channel)
    if img is None:
        raise ValueError("Could not find requested file inside 'pics' folder. ")
    if channel:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
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
                    [0.212, -0.523, 0.311]]#define the constant mat with constant values
    Ans_mat = np.zeros_like(imgRGB.astype(float)) # create new np.array of zeroes

    shape = imgRGB.shape[0]
    for val in range(shape):#move over all pixels in image
        Ans_mat[val, ...] = np.matmul(imgRGB[val, ...], constant_mat)
    return Ans_mat

    # yiq_ = np.array([[0.299, 0.587, 0.114],
    #                  [0.596, -0.275, -0.321],
    #                  [0.212, -0.523, 0.311]])
    # imYI = np.dot(imgRGB, yiq_.T.copy())
    # return imYI
    # mat = np.array([[0.299, 0.587, 0.144], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    # for x in imgRGB:
    #     for y in x:
    #         y[:] = mat.dot(y[:])
    # return imgRGB


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

    # mat = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]])
    # for x in imgYIQ:
    #     for y in x:
    #         y[:] = mat.dot(y[:])
    # return imgYIQ

    # rgb_ = np.array([[1.00, 0.956, 0.623],
    #                 [1.0, -0.272, -0.648],
    #                 [1.0, -1.105, 0.705]])
    # imRGB = np.dot(imgYIQ, rgb_.T.copy())
    # return imRGB

def Regular_histogram_Building(imgOrig: np.ndarray) -> (np.ndarray,int):
    """
    :param imgOrig: the array of an image
    :return: The array of a regular histogram of an image,
            and number represent the image dimension - 1 if grayscale and 2 if RGB
    """
    if imgOrig.ndim == 2:  # if this is gray scale
        # first , we will count the unique values of the array
        # because we get normalized values we will multiply each element in 255 and round it
        (unique, counts) = (np.unique(np.round(imgOrig * 255), return_counts=True))
        Ans_mat = np.zeros(256)  # create new np.array of zeroes in size of 256 because of the values [0,255]

        for i in range(len(counts)):
            Ans_mat[int(unique[i])] = counts[i]
            # example:
            # counts:    5 10 1 20 1
            # unique:    0 5 20 30 255
            # ans_mat:   5 0 0 0 0 10 0 0 ...1
            i = i + 1
        finalArr_beforeCumSum = Ans_mat
        return finalArr_beforeCumSum,1  # return the final array of regular histogram (1 if grayscale)

    elif imgOrig.ndim == 3:  # if this is RGB
        # first, as written in PDF, we will convert the RGB to YIQ image
        YIQimage = transformRGB2YIQ(imgOrig)  # convert the RGB array to YIQ
        Ychannel = YIQimage[:, :, 0]  # we will extract the first column as Y channel (those are normalized values between 0-1)
        # we want to turn the normalized values to values between 0-255,
        # therefore first we will multiply every element in 255
    # scaler_define = MinMaxScaler(feature_range=(0, 255))
    # scaledYchannel = scaler_define.fit_transform(Ychannel)
        # because those are float numbers, we will round them, and count the unique values of the array
        # (unique, counts) = np.unique(np.round(scaledYchannel), return_counts=True)
        Ychannel = cv2.normalize(Ychannel, None, 0, 255, cv2.NORM_MINMAX)
        Ychannel = np.ceil(Ychannel)  # floating point precision correction
        Ychannel = Ychannel.astype('uint8') # turn all number to int
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
        return finalArr_beforeCumSum,2  # return the final array of regular histogram (2 if RGB)
def Cumsum_histogram_Building(histArray: np.ndarray) -> (np.ndarray):
    """
    :param histArray: this is the array we got from the Regular_histogram_Building function
    :return: array of the cumsum
    """
    sum_mat = np.zeros(256)  # create new np.array of zeroes in size of 256 because of the values [0,255]
    i = 0
    for value in histArray:
        if i == 0:
            sum_mat[i] = value  # first element will be the same
        else:
            sum_mat[i] = sum_mat[i - 1] + value

        i = i + 1  # increase the index

    return sum_mat

def LUT_Building(SumArr: np.ndarray) -> (np.ndarray):
    """

    :param SumArr: The CumSum array of the image
    :return: Look Up Table as array
    """
    LUTarr = np.zeros(256)  # create new np.array of zeroes in size of 256 because of the values [0,255]
    for i in range(len(SumArr)):
        LUTarr[i]=math.ceil((SumArr[i]*255)/(SumArr[len(SumArr)-1])) #excactly the same formula as written in the PDF
    return LUTarr

def ApplyLUT(lutArr: np.ndarray,imOrig:np.ndarray) -> (np.ndarray):
    """
    :param lutArr: The lut array we created in LUT_Building
           imOrig: The array of the original image
    :return: the original array after we apply the LUT on it
    """
    tempimOrig=np.copy(imOrig)
    if imOrig.ndim == 2:  # if this is gray scale
        for i in range (len(tempimOrig)): # move over all rows
            for k in range (len(tempimOrig[0])): # move over all columns
                tempimOrig[i][k]=lutArr[int(np.round(imOrig[i][k]*255))]
        return tempimOrig


    elif imOrig.ndim == 3:  # if this is RGB
        imgEq= transformRGB2YIQ(imOrig) # we will transform it to YIQ (values between 0-1)
        imgEq[...,0]=(imgEq[...,0]-np.min(imgEq[...,0]))/(np.max(imgEq[...,0])-np.min(imgEq[...,0]))*255 #normlized Y channel to values 0-255
        for i in range(len(imgEq)): # move over all rows
            for k in range(len(imgEq[0])): # move over all columns
                newVal=lutArr[int(np.round(imgEq[i][k][0]))]
                imgEq[i][k][0]=newVal # we will put the value we want from LUT in the new array, in the right index
        #after applying the LUT , we get values in Ychannel between 0-255
        imgEq[...,0]=imgEq[...,0]/255 #we will normalized y channel to values between 0-1
        imgEq=transformYIQ2RGB(imgEq) # we will return it to RGB image
        plt.imshow(imgEq)
        return imgEq



def hsitogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    """
            Equalizes the histogram of an image
            :param imgOrig: Original Histogram
            :ret
        """
    if len(imOrig.shape) == 2:
        greyscale = True
    elif len(imOrig.shape) == 3:
        greyscale = False
    else:
        raise ValueError('Unsupported array representation. only RGB or Greyscale images allowed')
    tempImg = np.copy(imOrig)

    histOrig,ColorModel=Regular_histogram_Building(imOrig)    # Original Histogram
    cumSum=Cumsum_histogram_Building(histOrig)  #CumSum
    # plt.title('Original image histogram with CDF') #display them
    # plt.plot(cumSum, color='b')
    # plt.hist(histOrig.flatten(), 256, [0, 255], color='r')
    # plt.xlim([0, 255])
    # plt.legend(('cdf - ORIGINAL', 'histogram - ORIGINAL'), loc='upper left')
    lutArr = LUT_Building(cumSum)
    imgAfterLUT=ApplyLUT(lutArr,imOrig) #image equlized
    histEq,ColorModel=Regular_histogram_Building(imgAfterLUT)    # Original Histogram
    plt.subplot(2, 1, 2)
    plt.title('Equalized image histogram with CDF ')
    # plt.plot(cdf_normalized, color='b')
    # plt.hist(imEq.flatten(), 256, [0, 255], color='r')
    # plt.xlim([0, 255])
    # plt.legend(('cdf - EQUALIZED', 'histogram - EQUALIZED'), loc='upper right')
    # plt.show()

    # if ColorModel==1: #1 if grayscale
    #     pass
    # else:# and 2 if RGB
    #     pass

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
    if len(imOrig.shape) == 2:
        greyscale = True
    elif len(imOrig.shape) == 3:
        greyscale = False
    else:
        raise ValueError('Unsupported array representation. only RGB or Greyscale images allowed')

