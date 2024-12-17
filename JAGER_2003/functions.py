import numpy as np
import pandas as pd

def water_in_range(data, distance):
    """
    Function to determine the number of water pixels in a given distance range of a pixel
    :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
    :param distance: Number of pixels in each direction from the target pixel to consider
    :return: 2D numpy array with the number of water pixels in the given distance range of each pixel
    """

    # Create a 2D numpy array of zeros with the same shape as the input data
    water_in_range = np.zeros(data.shape)

    # Using a convolution iterate over the input data
    


    return water_in_range