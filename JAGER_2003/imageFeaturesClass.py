

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, distance_transform_edt
import pandas as pd

class ImageFeatures:
    """
    A class to represent and analyze image features related to water bodies.
    Attributes
    ----------
    path : str
        Path to the CSV file containing the image data.
    distance : int
        Number of pixels in each direction to consider for distance calculations.
    filename : str
        Name of the file without the extension.
    year : int
        Year extracted from the filename.
    path_next_year : str
        Path to the CSV file for the next year.
    area : str
        Area extracted from the filename.
    image : np.ndarray
        2D numpy array representing the image.
    image_next_year : np.ndarray
        2D numpy array representing the image for the next year.
    imagePadded : np.ndarray
        Padded version of the image with zeros.
    Methods
    -------
    get_year():
        Returns the year of the image.
    get_area():
        Returns the area of the image.
    get_image():
        Returns the image as a 2D numpy array.
    plot_image():
        Plots the image using matplotlib.
    euclidean_distance(x1, y1, x2, y2):
        Calculates the Euclidean distance between two points.
    angle_between_points(x1, y1, x2, y2):
        Calculates the angle between two points.
    amount_of_water_in_range():
        Determines the number of water pixels in a given distance range of each pixel.
    mask_water():
        Masks the water pixels in a given distance range of each pixel.
    find_possible_pixels(x, y):
        Finds possible water pixels within a given distance from a specified pixel.
    closest_water_pixel():
        Finds the closest water pixel for each pixel in the image.
    next_year_water(ind):
        Returns the water pixel value for the next year at the specified index.
    river_width():
        Calculates the river width for each pixel in the image.
    get_features():
        Extracts and returns various features related to water bodies in the image as a pandas DataFrame.
    """
    
    def __init__(self, path, distance):
        """
        Initializes the ImageFeatures class with the given path and distance.
        Parameters:
        path (str): The file path to the CSV file containing the image data.
        distance (int): The distance parameter used for padding the image.
        Attributes:
        path (str): The file path to the CSV file.
        distance (int): The distance parameter for padding.
        filename (str): The name of the file without the '.csv' extension.
        year (int): The year extracted from the filename.
        path_next_year (str): The file path for the next year's image data.
        area (str): The area extracted from the filename.
        image (ndarray): The image data loaded from the CSV file.
        image_next_year (ndarray): The image data for the next year loaded from the CSV file.
        imagePadded (ndarray): The image data padded with zeros based on the distance parameter.
        """
        
        self.path = path
        self.distance = distance
        self.filename = path.split('/')[-1].replace('.csv', '')
        self.year = int(self.filename.split('_')[1])
        self.path_next_year = self.path.replace(str(self.year), str(self.year + 1))
        self.area = self.filename.split('_')[3]
        self.image = np.genfromtxt(path, delimiter=',')
        self.image_next_year = np.genfromtxt(self.path_next_year, delimiter=',')
        self.imagePadded = np.pad(self.image, self.distance, mode='constant', constant_values=0)

    def get_year(self):
        """
        Retrieve the year attribute of the instance.
        Returns:
            int: The year associated with the instance.
        """
        return self.year
    
    def get_area(self):
        """
        Returns the area of the image feature.
        Returns:
            float: The area of the image feature.
        """
        return self.area
    
    def get_image(self):
        
        return self.image
    
    def plot_image(self):
        plt.imshow(self.image, cmap='gray')
        plt.title(f'Image {self.year} - {self.area}')
        plt.axis('off')
        plt.show()

    def euclidean_distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def angle_between_points(self, x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1)
    
    def amount_of_water_in_range(self):
        """
        Function to determine the number of water pixels in a given distance range of a pixel
        :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
        :param distance: Number of pixels in each direction from the target pixel to consider
        :return: 2D numpy array with the number of water pixels in the given distance range of each pixel
        """
        water = self.image == 1

        # Create a 2D numpy array of zeros with the same shape as the input data
        water_in_range = np.zeros(self.image.shape)

        # Define the kernel for the convolution
        kernel_size = 2 * self.distance + 1
        kernel = np.ones((kernel_size, kernel_size))
        kernel[self.distance, self.distance] = 0  # Exclude the center pixel

        # Perform the convolution using scipy.ndimage.convolve
        water_in_range = convolve(self.image, kernel, mode='constant', cval=0.0)
        water_in_range[water == 1] = 0  # Exclude the water pixels

        return water_in_range
    
    def mask_water(self):
        """
        Function to mask the water pixels in a given distance range of each pixel
        :param data: 2D numpy array of the image containing binary values (0 or 1) with 1 representing water and 0 representing land
        :param distance: Number of pixels in each direction from the target pixel to consider
        :return masked_data: 2D numpy array with the water pixels in the given distance range of each pixel masked
        :return water_in_range: 2D numpy array with the number of water pixels in the given distance range of each pixel
        """
        water_in_range = self.amount_of_water_in_range()
        masked_data = water_in_range > 0
        indices = np.argwhere(masked_data)
        return masked_data, water_in_range, indices
    
    def find_possible_pixels(self, x, y):
        padded_x = x + self.distance
        padded_y = y + self.distance
        possible_pixels = self.imagePadded[padded_x - self.distance: padded_x + self.distance + 1, padded_y - self.distance: padded_y + self.distance + 1]
        possible_ones = np.argwhere(possible_pixels == 1)
        x_coords = possible_ones[:, 0] + x - self.distance
        y_coords = possible_ones[:, 1] + y - self.distance
        return x_coords, y_coords

    def closest_water_pixel(self):
        masked_data, water_in_range, indices = self.mask_water()
        locations = []
        distances = []
        angles = []

        for i  in indices:
            possible_pixels = self.find_possible_pixels(*i)
            distances_i = [self.euclidean_distance(i[0], i[1], x, y) for x, y in zip(possible_pixels[0], possible_pixels[1])]
            closest_pixel_ind = np.argmin(distances_i)
            closest_pixel = (possible_pixels[0][closest_pixel_ind], possible_pixels[1][closest_pixel_ind])
            locations.append(closest_pixel)
            distances.append(distances_i[closest_pixel_ind])
            angles.append(self.angle_between_points(i[0], i[1], closest_pixel[0], closest_pixel[1]))
        
        water_in_range = water_in_range[indices[:, 0], indices[:, 1]]

        return indices, locations, distances, angles, water_in_range

    def next_year_water(self, ind):
        return self.image_next_year[ind[0], ind[1]]

    def river_width(self):
        edt, ind = distance_transform_edt(self.image, return_indices=True)
        river_width = np.zeros_like(self.image)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if edt[i, j] > river_width[ind[0, i, j], ind[1, i, j]]:
                    river_width[ind[0, i, j], ind[1, i, j]] = edt[i, j]
        river_width = river_width * 2
        return river_width



    def get_features(self):
        indices, locations, distances, angles, water_in_range = self.closest_water_pixel()

        next_year_water = [self.next_year_water(i) for i in indices]
        river_width = self.river_width()
        highest_values = []
        for i in locations:
            x, y = i
            x_min, x_max = max(0, x - 2), min(self.image.shape[0], x + 3)
            y_min, y_max = max(0, y - 2), min(self.image.shape[1], y + 3)
            highest_values.append(np.max(river_width[x_min:x_max, y_min:y_max]))


        df = pd.DataFrame({ 'year': [self.year] * len(indices), 'area': [self.area] * len(indices),
                            'index_x': indices[:, 0], 'index_y': indices[:, 1],
                            'distance': distances, 'sin_angle': np.sin(angles), 
                            'cos_angle': np.cos(angles), 'river_width': highest_values,
                            'water_in_range': water_in_range, 'next_year_water': next_year_water})
        return df

    

    
