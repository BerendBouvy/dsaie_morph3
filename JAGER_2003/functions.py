import numpy as np
import pandas as pd
from scipy.ndimage import convolve

def water_in_range(data, distance):
    """
    Function to determine the number of water pixels in a given distance range of a pixel
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param distance: Number of pixels in each direction from the target pixel to consider
    :return: 2D numpy array with the number of water pixels in the given distance range of each pixel
    """

    # Create a 2D numpy array of zeros with the same shape as the input data
    water_in_range = np.zeros(data.shape)

    # Define the kernel for the convolution
    kernel_size = 2 * distance + 1
    kernel = np.ones((kernel_size, kernel_size))
    kernel[distance, distance] = 0  # Exclude the center pixel

    # Perform the convolution using scipy.ndimage.convolve
    water_in_range = convolve(data, kernel, mode='constant', cval=0.0)

    return water_in_range

def distance_to_water(data, pixel_size):
    """
    Function to calculate the distance to the nearest water pixel for the center of each land pixel 
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param pixel_size: Size of each pixel in meters
    :return: 2D numpy array with the distance to the nearest water pixel for each land pixel
    """

    # Create a 2D numpy array of zeros with the same shape as the input data
    distance_to_water = np.zeros(data.shape)

    # Find the indices of the water pixels
    water_indices = np.argwhere(data == 1)

    # Loop over each land pixel
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:  # Check if the pixel is land
                # Calculate the distance to the nearest water pixel
                distances = np.linalg.norm(water_indices - np.array([i, j]), axis=1)
                min_distance = np.min(distances)
                # Convert the distance from pixels to meters
                distance_to_water[i, j] = min_distance * pixel_size

    return distance_to_water

