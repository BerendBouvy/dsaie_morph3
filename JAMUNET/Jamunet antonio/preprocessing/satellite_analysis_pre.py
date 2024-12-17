# This module contains the functions used for loading and preprocessing the satellite images

import os 
import math
import copy
import cv2 

import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from PIL import Image
from scipy.ndimage import rotate
from osgeo import gdal 

from preprocessing.images_analysis import show_image_array

# suppress the 'SettingWithCopyWarning' from pandas
pd.options.mode.chained_assignment = None

# set file directory
dir_orig = r'data\satellite\original' # original images
dir_proc = r'data\satellite\preprocessed' # preprocesed images

def count_pixels(image, pixel_value):
    """
    Count the number of pixels in an image with a specified value.

    Args:
        image (np.array): The input image array (2D array).
        pixel_value (int or float): The pixel value to count.

    Returns:
        int: The number of pixels equal to the specified pixel_value.
    """
    return np.sum(image == pixel_value)

def get_path_images(path, collection, train_val_test, year, month, day = 1, reach=1):
    '''
    Get the directory of the specific image given the collection, year, month, day, usage, and reach number of the image (*.tif format).
    
    Inputs:
           path = str, specifies directory where images are stored
           collection = str, specifies collection.
                        Available options: 'JRC/GSW1_4/MonthlyHistory' (the only one given here), 
                                           'LANDSAT/LT05/C02/T1_L2', 
                                           'COPERNICUS/S1_GRD'. 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation', and 'testing'
           year = int, year of image. Possible range: 1987-2024 for 'JRC/GSW1_4/MonthlyHistory', variable for the other datasets
           month = int, month of image. Possible range: 1-12
           day = int, day of image.
                 default: 1, otherwise possible range: 1-31
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
    
    Outputs: 
            file_path = str, full directory of the image file 
    '''
    # add 0 in front of month/day if smaller than 10 to match format
    if month < 10:
        month = f'0{month}'
    if day < 10:
        day = f'0{day}'
    
    if train_val_test is not None and reach is not None:
        file_path = path + fr'\{collection}_{train_val_test}_r{reach}\{year}_{month}_{day}_{train_val_test}_r{reach}.tif' 
    else:
        file_path = path + fr'\{collection}\{year}_{month}_{day}.tif' 
    
    return file_path

def rename_images(collection, train_val_test, reach=1, dir=r'data\satellite\original'):
    ''' 
    Rename the raw satellite image files for a given collection and usage of the images (training, validation, or testing) after being downloaded from Google Drive.
    Replace dashes with underscore digits and remove the collection name from the file name.

    Original name example: "JRC_GSW1_4_MonthlyHistory_1999-08-01_testing.tif"
    New name: "1999_08_01_testing.tif"

    Inputs:
           collection = str, specifies collection.
                        Available options: 'JRC/GSW1_4/MonthlyHistory' (the only one given here), 
                                           'LANDSAT/LT05/C02/T1_L2', 
                                           'COPERNICUS/S1_GRD'. 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
           dir = str, sets the directory where files are stored.
                 default: r'data\satellite\original', contains original (not preprocessed) images
                 other option: r'data\satellite\preprocessed' 
    
    Output: 
           none, rename all files by removing the collection name and replacing dashes with underscores
    '''
    # create path variable - join collection and images usage (training, validation, or testing)
    directory = os.path.join(dir, collection)
    if train_val_test is not None and reach is not None:
        path = directory + r'_' + train_val_test + fr'_r{reach}' 
    else:
        path = directory      
    
    # iterate over each file in the directory
    for filename in os.listdir(path):
        if filename.endswith('.tif'):

            if collection not in filename:
                raise Exception(f'There is no "{collection}" in the filenames: {filename}.\nCheck the directory, collection or use.') 

            # remove collection name from the file name
            _, date_with_extension = filename.split(collection + r'_', 1)
            
            # extract the date part and replace dashes with underscores
            date_without_extension = date_with_extension.split('.', 1)[0]
            new_filename = date_without_extension.replace('-', '_') + '.tif'

            # rename the file
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
            
    return None

def list_paths(collection, train_val_test, reach=1, dir_orig=r'data\satellite\original'):
    '''
    Create a list containing all the paths of the images stored in a specified folder.

    Inputs:
           collection = str, specifies collection.
                        Available options: 'JRC/GSW1_4/MonthlyHistory' (the only one given here), 
                                           'LANDSAT/LT05/C02/T1_L2', 
                                           'COPERNICUS/S1_GRD'. 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
           dir_orig = str, sets the directory where original files are stored.
                      default: r'data\satellite\original', contains original (not preprocessed) images
                      other option: r'data\satellite\preprocessed' 
    
    Output: 
           final_path = str, list of paths of the images given usage and reach
    '''
    # get directory given collection, use and reach
    collection_reach = fr'{collection}_{train_val_test}_r{reach}'

    # get full directory
    full_path = os.path.join(dir_orig, collection_reach)
    list = os.listdir(full_path)

    # generate final directory
    final_path = [os.path.join(full_path, list[i]) for i in range(len(list))]
    return final_path

def get_angle_rotation(image):
    '''
    Get the angle of rotation of the images before reshaping these.
    Automatically set the angle of counterclock-wise rotation based on the ratio between height and width of the input image.

    If height/width = 2 no rotation is applied (the flow direction in the given reach is already southward directed).
    If height/width = 1 a rotation of 45° is applied. 
    If height/width = 0.5 a rotation of 90° is applied.
    Otherwise an exception is raised.

    These angles are the result of the current input size of the images used for the deep-learning model. 
    Images should have a shape of (1000, 500) pixels and should ensure the flow is directed from the top to the bottom of the image. 
    If any change is done in the input images size this function requires to be modified.

    Inputs:
           image = np.array, representing GeoTIFF image with given shape
    
    Outputs:
            angle = int, angle of rotation of the input image. 
                    Possible values: 0°, 45°, 90°.
    '''
    image_shape = np.shape(image)
    ratio_h_w = round(image_shape[0]/image_shape[1], 2) # round the ratio to two decimal digits 

    if ratio_h_w == 2:
        angle = 0 # correct shape/orientation - no rotation needed
    elif ratio_h_w == 1:
        angle = 45
    elif ratio_h_w == 0.5:
        angle = 90
    else: 
        raise Exception(f'Image {image} has shape {image_shape} and height/width ratio = {ratio_h_w}, for which no angle is specified.')
    
    return angle

def rotate_images(image, reshape_img=True):
    '''
    Rotate the images such that the vertical orientation is presereved and the flow is directed from the top to the bottom of the image.
    the `rotate` function from `scipy.ndimage` library for the rigid rotation of the image is used.
    The angle of rotation is defined by the `get_angle_rotation` function implemented above. 

    Inputs:
           image = np.array, representing GeoTIFF image 
           reshape_img = bool, sets whether the rotation scipy rotation also reshapes the new rotated image (i.e., makes sure that the rotated image
                         size is large enough to contain the full oriringal image and avoiding to lose any pixel).
                         default: True, if set to False the new image is cropped/padded and some information might be lost.
    
    Outputs:
            rotated_image = np.array, rotated image with southward flow direction.
    '''
    angle = get_angle_rotation(image)
    rotated_image = rotate(image, angle, reshape=reshape_img, order=0) # order 0 to apply nearest-neighbor interpolation
    rotated_image = np.round(rotated_image, 0) # get only float integers
    return rotated_image

def reshape_images(image, desired_shape=(1000,500), new_padded_class=None):
    '''
    Reshape the images to the desired size before creating the input dataset for the model.
    If the original image shape is larger than the desired one, it crops the original image from both sides using the same cropping dimension. 
    If the original image shape is smaller than the desired one, it creates a padded image with assigned values (default is 0, but other values can be assigned) 
    and adds these on all sides of the image in an homogenoues way (i.e., it add the same amount of padded pixels on left and right side by checking the size mismatch 
    between current and desired width and does the same for the height).

    Inputs:
           image = np.array, representing GeoTIFF image with given shape
           desired_shape = list, contains the desired amount of pixels on both x- and y-direction 
                           default: (1000,500)
           new_padded_class = int, new class assigned to padded pixels if any.
                              default: None, padded pixels are assigned to `no-data` class (value=0).
                              Other options: any integer different from 1 and 2. Preferably either -1 or 3.
    
    Outputs:
            image_array = np.array, reshaped image with size equal to the desired one.
    '''
    current_shape = np.shape(image)

    if current_shape != desired_shape:
        # check height
        if current_shape[0] > desired_shape[0]:
            # get indeces to slice the array equally both sides
            top = (current_shape[0] - desired_shape[0]) // 2
            bottom = top + desired_shape[0]   
        else:
            top = 0
            bottom = current_shape[0]
        # check width
        if current_shape[1] > desired_shape[1]:
            # get indeces to slice the array equally both sides
            left = (current_shape[1] - desired_shape[1]) // 2
            right = left + desired_shape[1]
        else:
            left = 0
            right = current_shape[0]
    else:
        top = 0
        bottom = current_shape[0]
        left = 0
        right = current_shape[1]
    
    # new cropped image can still have a wrong size (if one or the other dimension is an odd number)
    cropped_image = image[top:bottom, left:right]
    
    # create a no-data image
    padded_image = np.zeros(desired_shape, dtype=image.dtype)
    if new_padded_class is not None:
        # replace pixel values if a new class is assigned to padded data
        padded_image[:,:] = new_padded_class

    # get sizes mismatch to ensure images are padded homogenously on both sides by the same amount of pixels 
    v_pad = (desired_shape[0] - cropped_image.shape[0]) // 2
    h_pad = (desired_shape[1] - cropped_image.shape[1]) // 2

    # place the cropped image into the padded image with the same amount of padded pixels on both sides, for each size 
    padded_image[v_pad:v_pad+cropped_image.shape[0], h_pad:h_pad+cropped_image.shape[1]] = cropped_image

    return padded_image

def preprocess_images(input_path, desired_shape=(1000,500), reshape_img=True, 
                      new_padded_class=None, input_f='original', output_f='preprocessed'):
    '''
    Preprocess the images applying the implemented rotation and reshaping algorithms.
    Create the new folders where the preprocessed images are stored.

    Inputs:
           input_path = str, path of the image to be preprocessed
           desired_shape = list, contains the desired amount of pixels on both x- and y-direction 
                           default: (1000,500)
           reshape_img = bool, sets whether the rotation scipy rotation also reshapes the new rotated image (i.e., makes sure that the rotated image
                         size is large enough to contain the full oriringal image and avoiding to lose any pixel).
                         default: True, if set to False the new image is cropped/padded and some information might be lost.
           new_padded_class = int, new class assigned to padded pixels if any.
                              default: None, padded pixels are assigned to `no-data` class (value=0).
                              Other options: any integer different from 1 and 2. Preferably either -1 or 3.
           input_f = str, specifies location of input folder.
                     default: 'original', if images that need to be preprocessed are stored in another directory this has to be changed.
           output_f = str, specifies location of output folder.
                      default: 'preprocessed', if images that need to be preprocessed will be stored in another directory this has to be changed.
    
    Outputs:
            none, preprocess images and save them in a new folder
    '''
    # load images
    img = gdal.Open(input_path)
    img_array = img.ReadAsArray()

    # rotate
    img_rot = rotate_images(img_array, reshape_img)
    # reshape
    img_resh = reshape_images(img_rot, desired_shape, new_padded_class)

    # get file name and output directory and path
    filename = os.path.basename(input_path)
    output_dir = os.path.dirname(input_path).replace(input_f, output_f)
    output_path = os.path.join(output_dir, filename)

    # create the output directory and save the file
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, img_resh)
    return None

def season_average(train_val_test, reach, year, dir_datasets=r'data\satellite', nodata=-1):
    '''
    Compute the average pixel values across a single low-flow season (from January to April of the same year).
    This is necessary for a later replacement of the `no-data` pixels, in order to ensure that images only have two classes
    (i.e., 0 = `non-water` and 1 = `water`).
    
    Initially replace `no-data` pixels with NaN value. Then average the images without taking into account these pixels.
    
    Inputs: 
           train_val_test = train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
           year = int, year of the season to average. Available range: 1988-2021 (for JRC collection)
           dir_datasets = str, specifies directory where original images are stored
                          default: r'data\satellite'
           nodata = int, represent pixel value for no-data class
                         default: -1, based on updated classes
                         If using the original classification, this should be set to 0

    Output:
           avg_season = 2D np.array, pixel-wise average values of images of the specified year low-flow season
    '''
    # initialize list
    imgs = []
    for month in range(1, 5):
        # get all images from low-flow season
        folder = os.path.join(dir_datasets, fr'dataset_month{month}')
        # get all reaches
        for reach_folder in os.listdir(folder):
            if reach_folder.endswith(f'{train_val_test}_r{reach}'):
                reach_path = os.path.join(folder, reach_folder)
                for image in os.listdir(reach_path):
                    if image.startswith(f'{year}'):
                        img_path = os.path.join(reach_path, image)
                        img = show_image_array(img_path, show=False)
                        img = np.where(img == nodata, np.nan, img) # replace nodata with nan
                        imgs.append(img)
    
    avg_season = np.nanmean(imgs, axis=0) # compute pixel-wise average excluding NaN
    return avg_season 

def replace_nan_with_neighbors_mean(image, window_size=15, replace_default=0):
    '''
    Compute the average of the `window_size` neighbors pixels to replace NaN values if still present.

    Inputs: 
           image = 2D np.array, season average image computed with the function `season_average`
           window_size = int, number of neighboring pixels around the NaN pixel, these are included in the average calculation
                         default: 15, as images have 60 m resolution and therefore the window is 60 x 15 = 900 m, close to the common
                         secondary channels width and to make sure that a large enouhg amount of pixels is included in the computation 
                         to avoid too many NaN pixels are included in the window (`no-data` pixels tend to lie close to each other) 
                         Pros: points in the middle of channels are surrounded by water, as well as points in non-water areas
                         Cons: high window implies the risk to miss the correct bank location, which is relevant for correct predictions
           replace_default = int, value used by default to replace any NaN pixel if still present after this averaging
                             default: 0, as `non-water` pixels for the scaled dataset
                             It has to be necessary an odd number, as the window around the NaN pixel is a square
    
    Output:
           neigh_image = 2D np.array, original image after NaN replacement with neighbors average
    '''
    # pad the image - add 1 line of NaN pixels on all sides
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=np.nan)
    rows, cols = image.shape

    neigh_image = np.copy(image)

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if np.isnan(padded_image[i, j]): # access only NaN pixels
                # extract the square window around the NaN pixel
                neighbors = padded_image[
                    i-(window_size//2):i+((window_size//2)+1), 
                    j-(window_size//2):j+((window_size//2)+1)
                                        ].flatten() # create a 1D array

                # calculate mean of valid (non-nan) neighbors
                valid_neighbors = neighbors[~np.isnan(neighbors)]
                if valid_neighbors.size > 0: # if there is at least one no NaN pixel in the window
                    mean_value = np.mean(valid_neighbors)
                else:
                    mean_value = replace_default  #  default value if all neighbors are NaN
                
                # replace NaN with the calculated mean
                neigh_image[i-1, j-1] = mean_value

    return neigh_image

def get_good_avg(train_val_test, reach, year, dir_datasets=r'data\satellite', nodata=-1, window_size=15, replace_default=0):
    '''
    Binarise images by replacing `no data` pixels with the average value across the season. If the average is still of `no-data` class,
    a neighboouring average value is taken and replaced. If the neighboour average also does not return a value because all pixels within the window
    are NaN pixels, a default value is then used.
    If there is the need to replace a single image, the last line before the return should be uncommented 

    Inputs: 
           train_val_test = train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
           year = int, year of the season to average. Available range: 1988-2021 (for JRC collection)
           dir_datasets = str, specifies directory where original images are stored
                          default: r'data\satellite'
           nodata = int, represent pixel value for no-data class
                         default: -1, based on updated classes
                         If using the original classification, this should be set to 0
           window_size = int, number of neighboring pixels around the NaN pixel, these are included in the average calculation
                         default: 15, as images have 60 m resolution and therefore the window is 60 x 15 = 900 m, close to the common
                         secondary channels width and to make sure that a large enouhg amount of pixels is included in the computation 
                         to avoid too many NaN pixels are included in the window (`no-data` pixels tend to lie close to each other) 
                         Pros: points in the middle of channels are surrounded by water, as well as points in non-water areas
                         Cons: high window implies the risk to miss the correct bank location, which is relevant for correct predictions
           replace_default = int, value used by default to replace any NaN pixel if still present after this averaging
                             default: 0, as `non-water` pixels for the scaled dataset
                             It has to be necessary an odd number, as the window around the NaN pixel is a square
    
    Output:
           good_avg = 2D np.array, final season average image with only two pixel classes and no NaN value. 
    '''
    avg = season_average(train_val_test, reach, year, dir_datasets, nodata)
    good_avg = replace_nan_with_neighbors_mean(avg, window_size, replace_default)
    good_avg = (good_avg > 0.5).astype(float) # classify as 1 if avg > 0.5, else as 0
    # if need to replace a single image uncomment the following line 
    # good_img = np.where(img == nodata, good_avg, img)
    return good_avg

def export_good_avg(train_val_test, reach, years=[1988, 2021], dir_datasets=r'data\satellite', 
                    dir_output=r'data\satellite\averages', nodata=-1, window_size=15, replace_default=0):
    '''
    Save the average images of a specific reach as *.csv files for later replacing the input images for the model

    Inputs: 
           train_val_test = train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
           years = list of ints, contains first and last years of the seasons to be averaged. 
                   default: [1988-2021]
           dir_datasets = str, specifies directory where original images are stored
                          default: r'data\satellite'
           dir_output = str, specifies directory where *.csv files are saved
                        default: r'data\satellite\averages'
           nodata = int, represent pixel value for no-data class
                         default: -1, based on updated classes
                         If using the original classification, this should be set to 0
           window_size = int, number of neighboring pixels around the NaN pixel, these are included in the average calculation
                         default: 15, as images have 60 m resolution and therefore the window is 60 x 15 = 900 m, close to the common
                         secondary channels width and to make sure that a large enouhg amount of pixels is included in the computation 
                         to avoid too many NaN pixels are included in the window (`no-data` pixels tend to lie close to each other) 
                         Pros: points in the middle of channels are surrounded by water, as well as points in non-water areas
                         Cons: high window implies the risk to miss the correct bank location, which is relevant for correct predictions
           replace_default = int, value used by default to replace any NaN pixel if still present after this averaging
                             default: 0, as `non-water` pixels for the scaled dataset
                             It has to be necessary an odd number, as the window around the NaN pixel is a square
    
    Output:
           none, saves the *.csv file in the specified folder
    ''' 
    # create list with years within given first and last years 
    years = np.arange(years[0], years[1]+1)
    folder_name = f'average_{train_val_test}_r{reach}'

    # create folder if not existing
    os.makedirs(os.path.join(dir_output, folder_name), exist_ok=True)

    for year in years:
       # compute average images 
        avg_img = get_good_avg(train_val_test, reach, year, dir_datasets, 
                               nodata, window_size, replace_default=replace_default)
        df = pd.DataFrame(avg_img)
        output_name = f'average_{year}_{train_val_test}_r{reach}.csv'
        output_path = os.path.join(dir_output, folder_name, output_name)
        # save average images 
        df.to_csv(output_path, index=False, header=False)
    return None

def load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages'):
    ''''
    Load the average image of a given year saved as *.csv file 

    Inputs: 
           train_val_test = train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
           year = int, year of the season to average. Available range: 1988-2021 (for JRC collection)
           dir_averages = str, specifies directory where *.csv files of the average images are stored
                          default: r'data\satellite\averages' 
    
    Output:
           img = 2D np.array image representing season average image of the given year 
    '''
    path = os.path.join(dir_averages, f'average_{train_val_test}_r{reach}', rf'average_{year}_{train_val_test}_r{reach}.csv')
    img = pd.read_csv(path, header=None).to_numpy()
    return img

def get_path(collection, cloud_cov=100, old=False):
    '''
    Get the path of the .csv file containing the number of satellite images available of a given dataset 
    based on the collection name

    Inputs: 
           collection = str, specifies collection name accordingly to Google Earth Engine nomenclature
           cloud_cov = int, specifies max percentage of cloud cover above which images are discarded. 
                       default: 100, if set to 'None' cloud coverage is undefined and all images are considered
           old = boolean, specifies whether old charts are loaded - a different name was given to the files
                 default: False, only loads newest data.  

    Outputs: 
            file_path = str, file path of the .csv file of the given collection.
    '''
    if cloud_cov == 'None':
         cloud_cov = '_undefined'
    file_path = fr'\total_images_cloud{cloud_cov}' + r'/' + collection.replace('/', '_')  + fr'_chart_data_cloud{cloud_cov}' + '.csv'
    if old==True:
         file_path = r'\old\ee-chart_' + collection.replace('/', '_') + '.csv'
    return file_path

def load_df(collection, cloud_cov=100, dir_files=r'data\satellite\original', print_total=True, old=False):
    '''
    Load the .csv file containing information on images available for the given dataset.

    Inputs: 
           collection = str, specifies collection name accordingly to Google Earth Engine nomenclature
           cloud_cov = int, specifies max percentage of cloud cover above which images are discarded. 
                       default: 100, if set to 'None' cloud coverage is undefined and all images are considered
           dir_files = str, directory of the general folder containing all .csv files
                       default: 'data\satellite\original' 
           print_total = bool, specifies whether to print total number of imagse available or not
                         default: True, set to False for plot functions
           old = boolean, specifies whether old charts are loaded - a different name was given to the files
                 default: False, only loads newest data. 

    Outputs: 
            df = pandas df, containing date of image, count of images per day and cumulative amount of images
            print statement specifying total number of images for the given dataset
    '''
    file_path = get_path(collection, cloud_cov, old)
    df = pd.read_csv(dir_files + file_path)
    df.columns = ['Date (yyyy-mm-dd)', 'Count']

    # convert to date type 
    df['Date (yyyy-mm-dd)'] = pd.to_datetime(df['Date (yyyy-mm-dd)'], format='%Y-%m-%d') 
    # keep only day-month-year format
    df['Date (yyyy-mm-dd)'] = df['Date (yyyy-mm-dd)'].dt.date  

    # get total number of images per dataset
    df['Cumulative count'] = 1
    
    for i in range(1, len(df)):                                                           
        df['Cumulative count'].iloc[i] = df['Cumulative count'].iloc[i] + df['Cumulative count'].iloc[i-1] 
        
        # get time interval between two consecutive images
        delta_t = (df.at[i, 'Date (yyyy-mm-dd)'] - df.at[i - 1, 'Date (yyyy-mm-dd)']).days
        df.at[i, 'Delta t'] = delta_t
        
        # raise count by 1 if more than one image is taken in a day
        if df['Date (yyyy-mm-dd)'].iloc[i] == df['Date (yyyy-mm-dd)'].iloc[i-1]:
              df['Count'].iloc[i] += 1
    
    # set first entry of time interval to 0 (would be NaN otherwise)
    df['Delta t'].iloc[0] = 0

    if print_total == True:
        print(f"Total number of images for {collection}: {df['Cumulative count'].iloc[-1]}\n")
    df
    return df

