
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import imageFeaturesClass as ifc
import os
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import KDTree
"""
This script visualizes various features of a satellite image using matplotlib.
The script performs the following tasks:
1. Loads a satellite image from a CSV file.
2. Calculates the amount of water in range for each pixel.
3. Finds the closest water pixel for each pixel and calculates the distance and angle to it.
4. Computes the Euclidean distance transform of the image.
5. Calculates the river width for each pixel.
The script then plots the following four subplots:
1. Amount of water in range.
2. Distance to the closest water pixel.
3. Angle to the closest water pixel.
4. River width.
Modules:
    - matplotlib.pyplot: For plotting the images.
    - matplotlib.colors: For creating custom colormaps.
    - numpy: For numerical operations.
    - imageFeaturesClass: Custom module for image feature extraction.
    - os: For interacting with the operating system.
    - scipy.ndimage: For computing the distance transform.
    - sklearn.neighbors: For building a KD-Tree.
Functions:
    - main: The main function that executes the script.
Usage:
    Run the script directly to generate the plots.
"""
if __name__ == "__main__":
    # Load the satellite image from a CSV file
    path = "data/satellite/averages/average_testing_r1/average_2020_testing_r1.csv"
    image = ifc.ImageFeatures(path, 7)
    grey_cmap = ListedColormap(['palegoldenrod', 'navy'])
    plt.subplot(1,4,1)
    # Calculate the amount of water in range for each pixel
    water_in_range = image.amount_of_water_in_range()
    ind , _, distance, angle, _ = image.closest_water_pixel()
    zero = np.zeros_like(image.get_image())
    zero[ind[:,0], ind[:,1]] = distance
    distance = zero
    zero = np.zeros_like(image.get_image())
    zero[ind[:,0], ind[:,1]] = angle
    angle = zero    
    edt, ind = distance_transform_edt(image.get_image(), return_indices=True)
    plt.imshow(water_in_range)
    plt.title("Amount of water in range", fontsize=15)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.imshow(distance)
    plt.title("Distance to closest water pixel", fontsize=15)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.title("Angle to closest water pixel", fontsize=15)
    plt.imshow(angle)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,4,4)
    
    rw = np.zeros_like(image.get_image())
    
    # Calculate the river width
    riverwidth = image.river_width()
    # Find indices where river width is not zero
    width_ind = np.nonzero(riverwidth)

    # Create an image to display non-zero river width locations
    river_width_image = np.zeros_like(riverwidth)
    river_ind = np.nonzero(image.get_image())
    
    set_A = np.vstack((river_ind[0], river_ind[1])).T
    set_B = np.vstack((width_ind[0], width_ind[1])).T
    
    # Build KD-Tree for set B
    tree = KDTree(set_B)

    # For each point in set A, find the closest point in set B
    distances, indices = tree.query(set_A, k=1)
    
    for i, edx in enumerate(indices):
        rw[river_ind[0][i], river_ind[1][i]] = riverwidth[width_ind[0][edx], width_ind[1][edx]]
    
    plt.subplot(1, 4, 4)
    plt.imshow(rw)
    plt.title("River width", fontsize=15)
    plt.axis('off')
    plt.colorbar()
    plt.show()
    
    

