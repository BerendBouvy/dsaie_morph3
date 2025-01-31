from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import imageFeaturesClass as ifc
import os
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import KDTree


if __name__ == "__main__":
    path = "data/satellite/averages/average_testing_r1/average_2020_testing_r1.csv"
    image = ifc.ImageFeatures(path, 7)
    grey_cmap = ListedColormap(['palegoldenrod', 'navy'])
    plt.subplot(1,4,1)
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
    
    

# # Example coordinates for set A and B
# x_coords_A = np.array([1, 3, 5])
# y_coords_A = np.array([2, 4, 6])
# x_coords_B = np.array([7, 9, 11])
# y_coords_B = np.array([8, 10, 12])

# # Combine x and y coordinates into 2D arrays (n, 2), where n is the number of points
# set_A = np.vstack((x_coords_A, y_coords_A)).T  # Transpose to get shape (n, 2)
# set_B = np.vstack((x_coords_B, y_coords_B)).T  # Transpose to get shape (n, 2)

# # Build KD-Tree for set B
# tree = KDTree(set_B)

# # For each point in set A, find the closest point in set B
# distances, indices = tree.query(set_A, k=1)

# # Print the closest points in set B
# for i, idx in enumerate(indices):
#     print(f"Point {set_A[i]} is closest to point {set_B[idx]} with distance {distances[i]}")
