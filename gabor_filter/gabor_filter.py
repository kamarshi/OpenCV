#!/usr/bin/python

import os

import sys

import numpy as np

import cv2

import matplotlib.pyplot as plt


# Gabor filtering object as an exercise in opencv
# and numpy coding



class gabor_filter(object):

    def __init__(self):
        self.filters = []


    def clear_filters(self):
        self.filters = []


    def rgb2gray(self, img):
        '''
        img is rows x cols x [r, g, b] in layout (matplotlib format)
        [TODO] Truncating luma output towards zero for now
        '''
        temp = np.dot(img, [0.299, 0.587, 0.144])

        return temp.astype(np.uint8)


    def bgr2gray(self, img):
        '''
        img is rows x cols x [b, g, r] in layout (opencv format)
        [TODO] Truncating luma output towards zero for now
        '''
        temp = np.dot(img, [0.144, 0.587, 0.299])

        return temp.astype(np.uint8)


    def generate_filters(self, ksize, sigma, count, lambd, gamma):
        '''
        ksize (pixels) - kernel size in pixels (should be odd)
        sigma (1,?)    - appears to be peak-to-peak size
        count          - is the number of filters oriented between 0 and pi
        lambd (1,?)    - relates to spatial frequency of filter
        gamma (0,1)    - extent of filter in the window - smaller means longer
        '''
        for theta in np.arange(0, np.pi, np.pi / count):
            # Compute filter kernels at diff angles
            kern = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)

            # Why are we doing this?
            kern /= 1.5*kern.sum( )

            self.filters.append(kern)


    def apply_filters(self, img):
        '''
        img is presumed to be a single channel luma image
        '''
        out_imgs = []

        for kern in self.filters:
            out_img = cv2.filter2D(img, cv2.CV_8UC1, kern)

            out_imgs.append(out_img)

        return out_imgs



def main():
    '''
    1.  Read in image
    2.  Extract luma part
    3.  Generate filters
    4.  Apply filters to image to generate list of output images
    5.  Display collage of output images using subplot
    '''
    img = cv2.imread('/home/vijaykam/Downloads/opencv-3.0.0/samples/data/baboon.jpg')

    if img == None:
        print 'Failed to read image'

        exit()

    gbf = gabor_filter( )

    l_img = gbf.bgr2gray(img)

    plt.set_cmap('gray')

    # gbf.generate_filters(31, 4.0, 8, 10.0, 0.5)
    gbf.generate_filters(31, 4.0, 16, 10.0, 0.5)

    p_imgs = []

    p_imgs = gbf.apply_filters(l_img)

    total = len(p_imgs)

    cols = 4

    rows = np.ceil(float(total)/cols)

    indx = 1

    print '%d:%d:%d' % (total, rows, cols)

    for img in p_imgs:
        plt.subplot(rows, cols, indx)

        plt.imshow(img)

        indx += 1

    plt.show()


 
if __name__ == '__main__':
    main()

