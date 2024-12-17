import numpy as np
import pandas as pd
from scipy.ndimage import convolve

def amount_of_water_in_range(data, distance):
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
    distance_water = np.zeros(data.shape)

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
                distance_water[i, j] = min_distance * pixel_size

    return distance_water

def angle_to_water(data):
    """
    Function to calculate the angle to the nearest water pixel for the center of each land pixel with regard to the North direction
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :return: 2D numpy array with the angle to the nearest water pixel for each land pixel
    """

    # Create a 2D numpy array of zeros with the same shape as the input data
    water_direction = np.zeros(data.shape)

    # Find the indices of the water pixels
    water_indices = np.argwhere(data == 1)

    # Loop over each land pixel to find the nearest water pixel
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:  # Check if the pixel is land
                # Calculate the vector from the land pixel to each water pixel
                vectors = water_indices - np.array([i, j])
                # Calculate the distance to each water pixel
                distances = np.linalg.norm(vectors, axis=1)
                # Find the index of the nearest water pixel
                nearest_index = np.argmin(distances)
                # Calculate the angle to the nearest water pixel
                angle = np.arctan2(vectors[nearest_index, 1], vectors[nearest_index, 0])
                # Convert the angle from radians to degrees
                angle_degrees = np.degrees(angle)
                # Adjust the angle to be in the range [0, 360]
                water_direction[i, j] = (angle_degrees + 360) % 360

    return water_direction

def width_of_river(data, water_direction, distance_water, water_in_range, pixel_size):
    """
    Functions which calculates the width of the river in meters extending from the closest water pixel to the land pixel 
    till it meets a new land pixel in the same direction.
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param water_direction: 2D numpy array with the angle to the nearest water pixel for each land pixel
    :param distance_water: 2D numpy array with the distance to the nearest water pixel for each land pixel
    :param water_in_range: 2D numpy array with the number of water pixels in the given distance range of each pixel
    :param pixel_size: Size of each pixel in meters
    :return: 2D numpy array with the width of the river in meters
    """

    # Find the indices of land pixels with water in range
    land_indices = np.argwhere(data == 0)

    # Create a 2D numpy array of zeros with the same shape as the input data
    river_width = np.zeros(data.shape)

    # Loop over each land pixel with water in range
    for i, j in land_indices:
        if water_in_range[i, j] > 0:

            # Calculate the direction of the water flow
            direction = water_direction[i, j]

            # Calculate the vector from the land pixel to the nearest water pixel
            vector = np.array([np.cos(np.radians(direction)), np.sin(np.radians(direction))])

            # Initialize the river width
            dx = 0
            dy = 0

            # Initialize the coordinates of the current pixel
            x, y = i, j

            # Loop until a new land pixel is encountered in the same direction
            while True:
                # Calculate the coordinates of the next pixel in the direction of the water flow
                x += vector[0]
                y += vector[1]
                # Check if the next pixel is within the image bounds
                if x < 0 or x >= data.shape[0] or y < 0 or y >= data.shape[1]:
                    break
                # Check if the integer of the 
                # Check if the next pixel is land
                if data[int(x), int(y)] == 0:
                    break
                # Increment the river width
                dx += vector[0]
                dy += vector[1] 

            # Convert the river width from pixels to meters
            river_width[i, j] = np.sqrt(dx**2 + dy**2) * pixel_size

    return river_width
