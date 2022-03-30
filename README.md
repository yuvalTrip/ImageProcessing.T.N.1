# ImageProcessing.T.N.1

The system used to run the task was Windows 10 PC, using PyCharm on python 3.8.

The files appeared are:
* Readme.md 
* *.jpg, *.png - all test images used
* ex1_main - the main class provided us
* ex1_utils - the functions that have been implemented as requested
* gamma.py - the gamma correction function was implemented
* gitignore - files to be ignored by the github, irrelevant to project
* Ex1.pdf - The pdf file of the assignment
* 
The functions implemented in the assignment:

* myID() - defines my ID number
* imReadAndConvert() - Reads an image, and returns the image converted as requested (in grayscale or RGB)
* imDisplay() - Reads an image as RGB or GRAY_SCALE and displays it (in grayscale or RGB)
* transformRGB2YIQ() - Converts an RGB image to YIQ color space
* transformYIQ2RGB() - Converts an YIQ image to RGB color space
* hsitogramEqualize() - Equalizes the histogram of an image
* Regular_histogram_Building() - Calculating and returning the histogram of the image
* Cumsum_histogram_Building() - Calculating and returning the cumulative histogram of the image
* LUT_Building-Compute the Look-Up Table by using the cumulative histogram
* ApplyLUT-Returning the original array after we apply the LUT on it
* Find_BestCenters- Finding and returning the best nQuant centers for quantize the image.
* fix_z-  Calculate new z, using the formula as we learned in lecture.
* fix_q- Calculate new q using wighted average on the histogram
* convertToImg- Executing the quantization to the original image 
* gammaDisplay() - GUI for gamma correction
