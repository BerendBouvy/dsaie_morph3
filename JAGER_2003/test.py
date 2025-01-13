import os

import numpy as np
import imageFeaturesClass as ifc
import matplotlib.pyplot as plt
import timeit


print(os.getcwd())
def test1():
    print("Test 1")
    # Test 1
    path = "data/satellite/averages/average_testing_r1/average_1999_testing_r1.csv"
    image = ifc.ImageFeatures(path, 7)
    print(image.get_year())
    print(image.get_area())
    image.plot_image()

def test2(): 
    image = ifc.ImageFeatures("JAGER_2003/temp_9999_test_abc.csv", 3)
    image.plot_image()
    indices, location, distance, angle = image.closest_water_pixel()
    zero = np.zeros_like(image.get_image())
    zero[indices[:,0], indices[:,1]] = distance
    plt.imshow(zero)
    plt.show()

def test3():
    image = ifc.ImageFeatures("JAGER_2003/temp_9999_test_abc.csv", 3)
    # print(image.find_possible_pixels(0, 4))
    # image.plot_image()
    image.closest_water_pixel()



def test4():
    path = "data/satellite/averages/average_testing_r1/average_2008_testing_r1.csv"
    image = ifc.ImageFeatures(path, 7)
    image.river_width()
    # start_time = timeit.default_timer()
    # image.get_features()
    # elapsed = timeit.default_timer() - start_time
    # print(f"test4 executed in {elapsed} seconds")

if __name__ == "__main__":
    # os.chdir(r"C:\Users\beren\Documents\AES\MSc\Data_and_AI\dsaie_morph3\data")
    # test1()
    # test2()
    # test3()
    test4()