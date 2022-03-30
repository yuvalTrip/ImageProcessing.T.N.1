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
import argparse

import numpy as np

from ex1_utils import LOAD_GRAY_SCALE
import cv2  # library of Python bindings designed to solve computer vision problems.

def on_trackbar(val):
    pass

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    Show = True
    img = cv2.imread(img_path)

    if rep == 1:  # if this is gray scale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    title_window = 'Gamma Correction'
    trackbar = 'Gamma Value'
    cv2.namedWindow(title_window)

    # I will let the user put values between 0-200
    cv2.createTrackbar(trackbar, title_window, 1,200, on_trackbar)

    while Show:  # the gamma will always correct itself as the user decide
        # return trackbar position
        gamma = cv2.getTrackbarPos(trackbar, title_window)

        # as it written in pdf we need the slider value be 0-2 with resolution 0.01
        gamma = gamma / 100  # we will fix the value we will get because we get values 0-200 so we will divide by 100

        new_img = img / 255.0
        g_array = np.full(new_img.shape, gamma) # we will fill the new array in 'gamma' values
        new_img = np.power(new_img, g_array) # 'new_img' array elements raised to powers from 'g_array' array, element-wise.

        cv2.imshow(title_window, new_img)
        k = cv2.waitKey(1000)  # wait
        # if we want to close it
        if k == 27:  # esc
            break
        if cv2.getWindowProperty(title_window, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()  # if we will stop

#https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
#https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
