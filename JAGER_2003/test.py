import os
import numpy as np
import imageFeaturesClass as ifc
import matplotlib.pyplot as plt
import timeit


# print(os.getcwd())
def test1():
    """
    A test function that performs the following operations:
    1. Prints "Test 1".
    2. Loads an image from a specified CSV file path.
    3. Prints the year of the image.
    4. Prints the area of the image.
    5. Plots the image.
    6. Displays the amount of water in the image within a specified range.
    Note:
        The function assumes the existence of an `ifc.ImageFeatures` class and `plt` for plotting.
    """
    
    print("Test 1")
    # Test 1
    path = "data/satellite/averages/average_testing_r1/average_2003_testing_r1.csv"
    image = ifc.ImageFeatures(path, 7)
    print(image.get_year())
    print(image.get_area())
    image.plot_image()
    plt.imshow(image.amount_of_water_in_range())

def test2(): 
    """
    This function performs the following operations:
    1. Creates an ImageFeatures object using a specified CSV file and a parameter.
    2. Plots the image using the plot_image method of the ImageFeatures object.
    3. Finds the closest water pixel and retrieves its indices, location, distance, and angle.
    4. Creates a zero matrix with the same shape as the image.
    5. Updates the zero matrix with the distances at the corresponding indices of the closest water pixels.
    6. Displays the updated zero matrix as an image using matplotlib.
    Note:
        - The CSV file path and the parameter for ImageFeatures are hardcoded.
        - The function assumes that the ImageFeatures class and necessary libraries (numpy, matplotlib) are imported.
    """
    
    image = ifc.ImageFeatures("JAGER_2003/temp_9999_test_abc.csv", 3)
    image.plot_image()
    indices, location, distance, angle = image.closest_water_pixel()
    zero = np.zeros_like(image.get_image())
    zero[indices[:,0], indices[:,1]] = distance
    plt.imshow(zero)
    plt.show()

def test3():
    """
    Test function to demonstrate the usage of the ImageFeatures class.
    This function performs the following steps:
    1. Creates an instance of the ImageFeatures class with a specified CSV file and a parameter.
    2. Calls the closest_water_pixel method on the image instance.
    Note: The find_possible_pixels and plot_image methods are commented out.
    Args:
        None
    Returns:
        None
    """
    
    image = ifc.ImageFeatures("JAGER_2003/temp_9999_test_abc.csv", 3)
    # print(image.find_possible_pixels(0, 4))
    # image.plot_image()
    image.closest_water_pixel()

def test4():
    """
    Test function to calculate river width from satellite image data.
    This function initializes an ImageFeatures object with a specified CSV file path
    and a parameter, then calls the river_width method on the object.
    The CSV file contains average testing data for the year 2008.
    Parameters:
    None
    Returns:
    None
    """
    path = "data/satellite/averages/average_testing_r1/average_2008_testing_r1.csv"
    image = ifc.ImageFeatures(path, 7)
    image.river_width()
    # start_time = timeit.default_timer()
    # image.get_features()
    # elapsed = timeit.default_timer() - start_time
    # print(f"test4 executed in {elapsed} seconds")


def test5():
    """
    This function performs a series of operations on satellite image data using the ImageFeatures class.
    The operations include:
    - Loading an image from a specified path.
    - Printing the area and year of the image.
    - Plotting the image.
    - Displaying the amount of water in range.
    - Creating and displaying a mask of water regions.
    - Displaying the water regions within a specified range.
    - Finding and displaying the closest water pixel distances and angles.
    - Calculating and displaying the river width.
    - Printing the extracted features of the image.
    The function uses matplotlib for plotting and numpy for array manipulations.
    """
    path = "data/satellite/averages/average_testing_r1/average_2008_testing_r1.csv"
    image = ifc.ImageFeatures(path, 7)
    print(image.get_area())
    print(image.get_year())
    image.plot_image()
    plt.imshow(image.amount_of_water_in_range())
    plt.title("Amount of water in range")
    plt.show()
    mask, water_in_range, indices = image.mask_water()
    plt.imshow(mask)
    plt.title("Mask")
    plt.show()
    plt.imshow(water_in_range)
    plt.title("Water in range")
    plt.show()
    indices, locations, distances, angles, water_in_range = image.closest_water_pixel()
    zero = np.zeros_like(image.get_image())
    zero[indices[:,0], indices[:,1]] = distances
    plt.imshow(zero)
    plt.title("Distances")
    plt.show()
    zero = np.zeros_like(image.get_image())
    zero[indices[:,0], indices[:,1]] = angles
    plt.imshow(zero)
    plt.title("Angles")
    plt.show()
    river_width = image.river_width()
    plt.imshow(river_width)
    plt.title("River width")
    plt.show()
    print(image.get_features())
    

if __name__ == "__main__":
    """ 
    Main function to test the ImageFeatures class and its methods.
    """
    # os.chdir(r"C:\Users\beren\Documents\AES\MSc\Data_and_AI\dsaie_morph3\data")
    # test1()
    # test2()
    # test3()
    # test4()
    test5()
    