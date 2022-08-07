"""
2. Matrix Manipulation

Write a program grayscale.py that converts a colored image into grayscale.

A color image is a width×height×3 matrix. Use any python package to read the
image as a matrix. Convert each pixel (with R,G,B values) to a single grayscale
value using the equation below.

    value = 0.2989R + 0.5870G + 0.1140B

An example code to read the image is:
    from matplotlib.image import imread
    image_rgb = imread(infile)

An example code to write the image is:
    from matplotlib import pyplot as plt
    #pyplot requires pixel values to be between 0 and 1
    image_gray = plt.imsave(’outfile.jpeg’,outfile/255)

"""

import cv2
import numpy as np

def rgb2gray(image):
    g = list()
    row,col,channel = image.shape
    for i in range(row) :
        for j in range(col):
            a = (image[i,j,0]*0.2989 + image[i,j,1]*0.5870 + image[i,j,2] *0.1140) 
            g.append(a)
    gr = np.array(g)
    return gr.reshape(row,col)

image_rgb = cv2.imread("images/image_rgb.jpg")
image_gray = rgb2gray(image_rgb)
cv2.imwrite("images/image_gray.jpg", image_gray)