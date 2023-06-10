#!/usr/bin/python3

# Implementation of SfM

import cv2 as cv

img = cv.imread('Dog_RGB/im*')
img1 = cv.resize(img, (600,600))
cv.imshow('image', img1)


cv.waitKey(0)
cv.destroyAllWindows()