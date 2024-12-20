import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from numpy import floor

def amount_of_water_in_range(data, distance):
    """
    Function to determine the number of water pixels in a given distance range of a pixel
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param distance: Number of pixels in each direction from the target pixel to consider
    :return: 2D numpy array with the number of water pixels in the given distance range of each pixel
    """
    water = data == 1

    # Create a 2D numpy array of zeros with the same shape as the input data
    water_in_range = np.zeros(data.shape)

    # Define the kernel for the convolution
    kernel_size = 2 * distance + 1
    kernel = np.ones((kernel_size, kernel_size))
    kernel[distance, distance] = 0  # Exclude the center pixel

    # Perform the convolution using scipy.ndimage.convolve
    water_in_range = convolve(data, kernel, mode='constant', cval=0.0)
    water_in_range[water == 1] = 0  # Exclude the water pixels

    return water_in_range


def mask_water(data, distance):
    """
    Function to mask the water pixels in a given distance range of each pixel
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param distance: Number of pixels in each direction from the target pixel to consider
    :return masked_data: 2D numpy array with the water pixels in the given distance range of each pixel masked
    :return water_in_range: 2D numpy array with the number of water pixels in the given distance range of each pixel
    """
    water_in_range = amount_of_water_in_range(data, distance)
    masked_data = water_in_range > 0
    indices = np.argwhere(masked_data)
    return masked_data, water_in_range, indices

def distance_to_water(data, pixel_size):
    """
    Function to calculate the distance to the nearest water pixel for the center of each land pixel 
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param pixel_size: Size of each pixel in meters
    :return distance_water: 2D numpy array with the distance to the nearest water pixel for each land pixel
    :return nearest_water_pixel: 3D numpy array with indices of the nearest water pixel for each land pixel
    """

    # Create a 2D numpy array of zeros with the same shape as the input data
    distance_water = np.zeros(data.shape)

    # Create a 3D numpy array to store the indices of the nearest water pixel for each land pixel
    nearest_water_pixel = np.zeros((data.shape[0], data.shape[1], 2), dtype=int)

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

                # Find the indices of the nearest water pixels
                nearest_indices = np.argwhere(distances == min_distance)
                nearest_water_pixel[i,j] = water_indices[nearest_indices[0],:]


    return distance_water, nearest_water_pixel

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
                # Store the angle in the water_direction array
                water_direction[i, j] = angle_degrees

    return water_direction

def width_of_river(data, water_direction, nearest_water_pixel, water_in_range, pixel_size):
    """
    Functions which calculates the width of the river in meters extending from the closest water pixel to the land pixel 
    till it meets a new land pixel in the same direction.
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param water_direction: 2D numpy array with the angle to the nearest water pixel for each land pixel
    :param nearest_water_pixel: 3D numpy array with the indices of the nearest water pixel for each land pixel
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

            # Initialize the coordinates of the water source closest to the current pixel
            x, y = nearest_water_pixel[i, j, 0], nearest_water_pixel[i, j, 1]

            # Store the coordinates of the current pixel
            x_start, y_start = x, y

            # Set the center of the pixel as the starting position
            x = x+0.5
            y = y+0.5

            # Loop until a new land pixel is encountered in the same direction
            while True:
                # Calculate the coordinates of the next pixel in the direction of the water flow
                x_prev, y_prev = x, y
                x += vector[0]
                y += vector[1]
                # Check if the next pixel is within the image bounds
                if x < 0 or x >= data.shape[0] or y < 0 or y >= data.shape[1]:
                    break

                # Check if the integer of the coordinates has changed
                elif int(floor(x)) != int(floor(x_prev)) or int(floor(y)) != int(floor(y_prev)):
                    # Check if the next pixel is land
                    if data[int(floor(x)), int(floor(y))] == 0:
                        break
                else:
                    continue
            
            # Coordinates of the new land pixel in the same direction
            x_end, y_end = floor(x), floor(y)

            # Convert the river width from pixels to meters
            river_width[i, j] = np.sqrt((x_end-x_start)**2 + (y_end-y_start)**2) * pixel_size

    return river_width