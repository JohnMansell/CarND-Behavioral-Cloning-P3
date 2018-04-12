'''---------------------------------------
        Import Statements
---------------------------------------'''

import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import random
from tempfile import TemporaryFile


correction = 0.25
num_bins = 23
colorConversion = cv2.COLOR_BGR2LAB

'''---------------------------------------
        Read data from File
---------------------------------------'''
def read_data_from_file(fileName, lineArray):
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lineArray.append(line)

'''---------------------------------------
        Extract images and Measurements
---------------------------------------'''


def get_images_and_measurements(lineArray, splitToken, imagePath, imageArray, measurementArray):
    for line in lineArray:
        for i in range(3):
            source_path = line[i]
            tokens = source_path.split(splitToken)
            filename = tokens[-1]
            local_path = imagePath + filename
            image = cv2.imread(local_path)
            imageArray.append(image)
        measurement = float(line[3])
        measurementArray.append(measurement)
        measurementArray.append(measurement + correction)
        measurementArray.append(measurement - correction)

'''---------------------------------------
        Print Histogram of Data
---------------------------------------'''
def print_histogram(measurement_array, show, title = ''):
    avg_samples_per_bin = len(measurement_array)/num_bins
    hist, bins = np.histogram(measurement_array, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(measurement_array), np.max(measurement_array)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    if show:
        plt.title(title)
        plt.show()

'''---------------------------------------
        Flip each image and measurement
---------------------------------------'''


def flip_image_and_measurement(imageArray, measurementArray, augmented_images, augmented_measurements):
    for image, measurement in zip(imageArray, measurementArray):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = measurement * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)

'''---------------------------------------
        Get Transform
---------------------------------------'''
def get_transform(img,
                    x_bottom = 1136,
                    x_top = 267,
                    depth = 83,
                    hood_depth = 33,
                    dst_offset = 271,
                    cal1_offset = 27,
                    cal2_offset = 30):

    img_size = (img.shape[1], img.shape[0])

    # src = (x1, y1) , (x2, y2), (x3, y3), (x4, y4)
    x1 = int((img_size[0] - x_top) / 2)
    x2 = int((img_size[0] + x_top) / 2)

    y1 = y2 = int((img_size[1] - depth))

    x3 = int((img_size[0] - x_bottom) / 2)
    x4 = int((img_size[0] + x_bottom) / 2)

    y3 = y4 = (img_size[1] - hood_depth)

    # dst = (j1, k1), (j2, k2), (j3, k3), (j4, k4)
    j1 = j3 = (img_size[0] / 2) - dst_offset
    j2 = j4 = (img_size[0] / 2) + dst_offset

    k1 = k2 = 0
    k3 = k4 = img_size[1]

    src = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    dst = np.float32([[j1, k1], [j2, k2], [j3, k3], [j4, k4]])

    # Perspective Transform -- Matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return (M, Minv)

'''---------------------------------------
        Warp
---------------------------------------'''
img = cv2.imread('myData/IMG/center_2018_02_26_12_32_17_315.jpg')
M, inv = get_transform(img)

def warp_image(img, mtx):
    img_size = (img.shape[1], img.shape[0])

    # Perspective Transform
    warped = cv2.warpPerspective(img, mtx, img_size, flags=cv2.INTER_LINEAR)

    return warped

'''----------------------------
        Mag Threshold
-------------------------------'''
smin = 3
smax = 255
bmin = 0
bmax = 209
dmin = 0.1
dmax = 0.9
m_min = 5
m_max = 311
d_kernal = 13
m_kernal = 5
picture = 5
sigma_color = 75
sigma_space = 75

def mag_threshold(image, sobel_kernel=m_kernal, mag_thresh = (m_min, m_max)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    binary_output = np.zeros_like(gradmag)

    binary_output[(gradmag >= mag_thresh[0])& (gradmag <= mag_thresh[1])] = 1
    return binary_output

'''----------------------------
        Color
-------------------------------'''
def color(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


'''----------------------------
        Create Binary
-------------------------------'''
def create_binary(img):

    warp = warp_image(img, M)

    bilateral = cv2.bilateralFilter(warp, m_kernal, sigma_color, sigma_space)

    mag = mag_threshold(bilateral, m_kernal, (m_min, m_max))

    result = np.copy(warp)
    result[(mag == 1)] = 0

    return result
'''---------------------------------------
        Balance DataSet
---------------------------------------'''
def balance_data_set(augmented_images, augmented_measurements, hist, bins, averageHeight, newImages, newMeasurements, lowerLimit, upperLimit):
    for image, measurement in zip(augmented_images, augmented_measurements):
        if (measurement < lowerLimit or measurement > upperLimit):
            for i in range(num_bins):
                if bins[i] < measurement < bins[i + 1]:
                    difference = abs(averageHeight - hist[i])
                    multiples = int(difference / hist[i])
                    for k in range(multiples):
                        brightness = random.randint(0, 100)
                        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                        y, u, v = cv2.split(yuv)
                        y -= brightness
                        final_yuv = cv2.merge((y, u, v))
                        newImage = cv2.cvtColor(final_yuv, cv2.COLOR_YUV2BGR)
                        newImages.append(newImage)
                        newMeasurements.append(measurement)


'''---------------------------------------
        PreProcess Image
---------------------------------------'''
def preprocess_image(img):
    img = create_binary(img)
    return img