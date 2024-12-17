# This module contains the functions used for the data analysis of the satellite images

import os
import shutil
import torch 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from osgeo import gdal 
from datetime import datetime
from PIL import Image, ImageOps

# set file directories
dir_orig = r'data\satellite\original' # original images
dir_proc = r'data\satellite\preprocessed' # preprocesed images

def load_image(file_path, tensor=True, show=False, cmap='gray', vmin=-1, vmax=1):
    '''
    Load and possibly show a single image using Image library. It also converts and returns it into a numpy array or a torch.tensor.
    By default it shows a grayscale image. It is implemented and tested to work with JRC collection exported in grayscale.

    It also scales pixels values to be in the following way:
        - no-data: -1
        - non-water: 0
        - water: 1

    Inputs: 
           file_path = str, contains full path of the image to be shown
           tensor = bool, sets whether the function returns a torch.tensor or a numpy.array
           show = bool, specifies whether image is shown or not.
                  default: False, if set to True image is shown
           cmap = str, key to set the visualization channels
                  default: 'gray'
           vmin = int, minimum value used for visualization.
                  default: -1, can take any value. 
                  If set to 0, no-data pixels will not be displayed 
           vmax = int, maximum value used for visualization.
                  default: 1, can take any value.
    
    Output: 
           torch.tensor(array_img) = torch.tensor, 2D tensor representing the loaded image
        or
           array_img = np.array, 2D array representing the loaded image           
    '''
    with Image.open(file_path) as img:
        img = img.convert('L')
        img = ImageOps.autocontrast(img, cutoff=(-1, 1))
        
        # convert to numpy.array
        array_img = np.array(img).astype(int)

        # scale pixel classes
        array_img[array_img == 0] = -1      # no-data
        array_img[array_img == 127] = 0     # non-water  
        array_img[array_img == 255] = 1     # water
        
        # show the image
        if show==True:
            plt.imshow(array_img, cmap=cmap, vmin=vmin, vmax=vmax)
            # get year, month and day of image by splitting the path
            year, month, day = file_path.split('\\')[-1].split('_')[:3]
            plt.title(f'{year}-{month}-{day}')
            plt.show()

    return torch.tensor(array_img) if tensor else array_img

def show_image_array(path, scaled_classes=True, cmap='gray', vmin=0, vmax=2, show=True, save_img=False):
    '''
    This function is used to load and show a single image using Gdal library. It also converts and returns it into a numpy array with dtype = np.float32.
    By default it shows a grayscale image. It is implemented and tested to work with JRC collection exported in grayscale (i.e., with pixel values between 0, 1 and 2).

    It also updates the pixels values by subtracting 1 from the whole image (element-wise operation) in order to later implement an algorithm that
    masks only non-water and water pixels (0 and 1 pixels, respectively) to train the model on these pixels only and neglect the no-data ones. 

    It can also scale the original pixel values by setting the new classes as follows:
            - no-data: -1
            - non-water: 0
            - water: 1

    Inputs: 
           path = str, contains full path of the image to be shown
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled. 
           cmap = str, key to set the visualization channels
                  default: 'gray'
           vmin = int, minimum value needed for visualization.
                  default: 0, can range from 0 to 255.  
           vmax = int, maximum value needed for visualization.
                  default: 1, can range from 0 to 255. 
           show = bool, specifies whether image is shown or not.
                  default: True, if set to False image is not shown
           save_img = bool, sets whether the function is used for saving the image or not. If used for for this, returns fig and ax too
                      default: False, image is not being saved
    
    Output: 
           img_array = np.array, 2D array representing the loaded image
           
    If save_img = True also returns
           fig = matplotlib.figure.Figure object, needed for saving function
           ax = matplotlib.axes._axes.Axes object, needed for saving function
    '''
    img = gdal.Open(path)
    img_array = img.ReadAsArray().astype(np.float32)

    # scale the pixel value for each class with the updated classification and change vmin and vmax
    if scaled_classes:

       img_array = img_array.astype(int)
       img_array[img_array==0] = -1
       img_array[img_array==1] = 0
       img_array[img_array==2] = 1
       vmin = -1
       vmax = 1

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img_array, cmap=cmap, vmin=vmin, vmax=vmax)
    # get year, month and day of image by splitting the path
    year, month, day = path.split('\\')[-1].split('_')[:3]
    ax.arrow(100, 125, 0, 200, width=10, facecolor='black', edgecolor='white')
    ax.set_title(f'{year}-{month}-{day}')
    ax.axis('on')  # ensure the axis is on
    
    shp = img_array.shape
    x_ticks = np.arange(0, shp[1]+1, 150)
    y_ticks = np.arange(0, shp[0]+1, 200)  

    # Convert x_ticks and y_ticks from pixels to meters
    x_tick_labels = [round(tick * 60/1000, 2) for tick in x_ticks]  
    y_tick_labels = [round(tick * 60/1000, 2) for tick in y_ticks]

    ax.set_xticks(x_ticks, x_tick_labels)
    ax.set_xlabel('Width (km)', fontsize=11)
    # ax.invert_yaxis() # can be confusing as flow is south-ward directed
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel('Lentgh (km)', fontsize=11) 

    if show:
       plt.show()

    return img_array, fig, ax if save_img else img_array

def load_all_csv(train_val_test, cols=['Date image', 'no-data: 0', 'non-water: 1', 'water: 2'], dir_folders=r'data\satellite\preprocessed', index_col=None):
    '''
    This function is used to load all `*.csv` files containing info on the pixel classes distribution of JRC collection images.
    It creates a list of dataframes, one for every reach.

    Inputs: 
           train_val_test = str, specifies for what the images are used for.
                           available options: 'training', 'validation' and 'testing'
           cols = list of str, contains the name of columns to be loaded 
                  default: ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2']
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\preprocessed'
           index_col = int or None, sets the column which is used as index for the new dataframe.
                       default: None, no column used as index. Index becomes the total count of images.
                       Other available options: 1, if 'Date image' column is used as index. Not recommended to avoid confusion 
    
    Output: 
           dfs = list, contains one dataframe for each river reach given the use (training, validation, testing)
    ''' 
    reach_id = 0
    dfs = [] # initiate list of dataframes

    folders = []

    for folder in os.listdir(dir_folders):
        if train_val_test in folder: # include only specific training, validation or testing reaches
            folders.append(folder)

    # sort folders based on reach_id
    folders.sort(key=lambda x: int(x.split(f'_{train_val_test}_r')[-1]))

    # load each single csv separately
    for folder in folders:
        folders_path = os.path.join(dir_folders, folder)
        for filename in os.listdir(folders_path):
            if filename.endswith('allpixels.csv'): 
                reach_id += 1
                path = os.path.join(folders_path, train_val_test + f'_r{reach_id}_' + 'allpixels.csv')
                if index_col == None:
                    df = pd.read_csv(path, usecols=cols) 
                else:
                    df = pd.read_csv(path, usecols=cols, index_col=index_col)
                dfs.append(df)
    return dfs

def create_long_df(train_val_test, cols=['Date image', 'no-data: 0', 'non-water: 1', 'water: 2'], dir_folders=r'data\satellite\preprocessed'):
    '''
    This function is used to create a complete dataframe for a given image use (training, validation and testing) with all reaches. 
    It contains the complete info on the pixel classes distribution of JRC collection images.
    It creates one full dataframe containing the values for every reach.

    Inputs: 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           cols = list of str, contains the name of columns to be loaded 
                  default: ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2']
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\preprocessed'
    
    Output: 
           df = dataframe, contains all pixel classes values among years and reaches given the use (training, validation and testing)
    ''' 
    # get list of dataframes
    dfs = load_all_csv(train_val_test, cols, dir_folders)

    # create dataframe with all values of training reach
    date_reach_col = []
    no_data_col = []
    non_water_col = []
    water_col = []

    for i in range(len(dfs)):
        df = dfs[i] # get single dataframe
        df.columns = cols

        # get only values of each column
        date_reach = df.iloc[:,0].values
        no_data = df.iloc[:,1].values
        non_water = df.iloc[:,2].values
        water = df.iloc[:,3].values

        # extend original empty list
        date_reach_col.extend(date_reach)
        no_data_col.extend(no_data)
        non_water_col.extend(non_water)
        water_col.extend(water)
    
    # zip lists to create complete dataframe
    all_reaches = list(zip(date_reach_col, no_data_col, non_water_col, water_col))
    df = pd.DataFrame(all_reaches)
    df.columns = cols
    return df

def get_info_images(df, acc_percentage, shape=(1000,500)):
    '''
    This function returns the number of available, perfect and unavailable images of a given dataframe 
    containing the total number of pixels for each class (0: no-data, 1: non-water, 2: water).

    It assumes that the dataframe has the following columns:
    ['no-data: 0', 'non-water: 1', 'water: 2']

    Inputs: 
            df = pandas dataframe, contains amount of pixels per each class given image and usage (training, validation and testing).
            acc_percentage = float, percentage of acceptable cloud cover (i.e., no-data pixels) over the total.
                             Acceptable values: 0 < acc_percentage < 1
            shape = tuple, specifies shape of the images, needed for calculating the total number of pixels.
                    default: (1000,500)
    Output: 
           none, prints a statement including relevant information on images availability
    '''
    pixels = shape[0]*shape[1]
    cloud_thr = int(pixels * acc_percentage)

    nodata = len(df[df['no-data: 0'] == pixels])
    perfect = len(df[df['no-data: 0'] == 0])
    ok = len(df[df['no-data: 0'] <= cloud_thr])

    print(f'Total images: {len(df)}.\n\
Completely cloudy images: {nodata}.\n\
Perfect (no clouds) images: {perfect}.\n\
Images usable (no-data pixels threshold = {cloud_thr}): {ok}.')
    return None

def clear_full_df(train_val_test, nodata_thr, water_thr, cols=['Date image', 'no-data: 0', 'non-water: 1', 'water: 2'], 
             dir_folders = r'data\satellite\preprocessed', dir_output = r'cleared_csv'):
    '''
    This function is used to select all images that respect the given amount of pixels data threshold (no-data max value and water min value).
    It slices through the input dataframe and gets rid of all images that do not comply with the requirements.
    It also saves a .csv file to store the clear dataframe with the information of the good images.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                           available options: 'training', 'validation' and 'testing'
           nodata_thr = int, max acceptable amount of no-data pixels 
           water_thr = int, max acceptable amount of water pixels
           cols = list of str, contains the name of columns to be loaded 
                  default: ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2']
           dir_folders = str, dirtectory where the original .csv files are stored
                        default: r'data\satellite\preprocessed'
           dir_output = str, contains directory where the new .csv files gets saved 
                        default: r'cleared_csv'

    Output:
           clean_df = pandas df, original dataframe without images that do not meet the set requirements         
    '''
    orig_df = create_long_df(train_val_test, cols, dir_folders)
    # select only images that meet criteria
    clean_df = orig_df[(orig_df['no-data: 0'] <= nodata_thr) & (orig_df['no-data: 0'] >= water_thr)]

    # save file
    output_file = fr'{train_val_test}_cleared_nodata{str(nodata_thr)}_water{str(water_thr)}.csv'
    output_path = os.path.join(dir_folders, dir_output, output_file)
    
    clean_df.to_csv(output_path)
    return clean_df

def clear_single_df(train_val_test, nodata_thr, water_thr, dir_folders = r'data\satellite\preprocessed'):
    '''
    This function removes the images that do not meet the requirements (no-data and water max/min pixels) from the original dataframe, which contains all images.
    It then saves the remaining images on another *.csv file in the same directory.

    This function performs the same operation of `clear_full_df` but compared to this one it accesses each *.csv file separately without creating a full one,
    i.e., it creates a file for each combination of use (training, validation and testing) and reach id.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                           available options: 'training', 'validation' and 'testing'
           nodata_thr = int, max acceptable amount of no-data pixels 
           water_thr = int, max acceptable amount of water pixels
           dir_folders = str, dirtectory where the original .csv files are stored
                        default: r'data\satellite\preprocessed'
    
    Output: 
           none, it creates a *.csv file with information on the available images given no-data and water thresholds 
    '''
    reach_id = 0
    folders = []

    for folder in os.listdir(dir_folders):
        if train_val_test in folder: # include only specific training, validation or testing reaches
            folders.append(folder)
    
    # sort folders based on reach_id
    folders.sort(key=lambda x: int(x.split(f'_{train_val_test}_r')[-1]))

    # loop through folders
    for folder in folders:
        folders_path = os.path.join(dir_folders, folder)
        for filename in os.listdir(folders_path):
            if filename.endswith('allpixels.csv'):
                reach_id += 1
                path = os.path.join(folders_path, train_val_test + f'_r{reach_id}_' + 'allpixels.csv')
                df = pd.read_csv(path)

                # get only images that meet the requirements
                clean_df = df[(df['no-data: 0'] <= nodata_thr) & (df['no-data: 0'] >= water_thr)]

                # save file
                output_file = fr'{train_val_test}_r{reach_id}_cleared_nodata{str(nodata_thr)}_water{str(water_thr)}.csv'
                output_path = os.path.join(dir_folders, folder, output_file)
                clean_df.to_csv(output_path)
    return None

def compute_diff(train_val_test, reach_id, dir_folders = r'data\satellite\preprocessed', collection = r'JRC_GSW1_4_MonthlyHistory'):
    '''
    This function loads all images of a specified folder (use and reach id), computes the difference between one image and the following one 
    considering only those values that are either -1 or 1 and counts all these different pixels and returns a dataframe containing for each combination 
    of start/end image the related pixels difference.  
    
    It only considers pixels whose value is either -1 and 1 because these result from the either a difference of [1-2] or [2-1], 
    which means there was erosion or sedimentaion, respectively. Other changes ([2-0], [0-2]) or non-changes ([0-0], [1-1] and [2-2]).

    The function is implemented and tested to work for grayscale images retrieved from JRC collection.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                           available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           dir_folders = str, dirtectory where the original .csv files are stored
                         default: r'data\satellite\preprocessed'
           collection = str, specifies the satellite images collection.
                       default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
    Output: 
           df = pandas DataFrame, contains info on pixels difference for each combination of start/end image
                columns: 'Start date', 'End date', 'Different pixels'  
    '''

    folder_path = os.path.join(dir_folders, collection + fr'_{train_val_test}_r{reach_id}')

    # initialize lists with dates and images
    names_date = []
    imgs = []

    # load each single tif image separately
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            name_date = filename.split(f'_{train_val_test}', 1)[0]
            names_date.append(name_date)
            img = show_image_array(path=file_path, show=False)
            imgs.append(img)
    
    # initialize total_difference, (interesting) difference and start/end date lists
    total_diffs = []
    diffs = []
    start_date = []
    end_date = []

    for i in range(1, len(imgs)):
        # simple diffeerence
        total_diff = imgs[i] - imgs[i-1]
        total_diffs.append(total_diff)
        
        # counts only total_diff = 1
        ones = np.count_nonzero(total_diff == 1) 
        # counts only total_diff = -1
        minus_ones = np.count_nonzero(total_diff == -1) 

        # gets sum of (-1,+1) changed pixels
        diff = ones + minus_ones
        diffs.append(diff)

        # gets start/end date combinations
        start_date.append(names_date[i-1])
        end_date.append(names_date[i])
        
    # create df
    combined = list(zip(start_date, end_date, diffs))
    df = pd.DataFrame(combined)
    df.columns = ['Start date', 'End date', 'Different pixels']
    return df

def convert_to_date(string_date):
    '''
    This function converts strings representing dates into datetime objects with format yyyy-mm-dd.
    It works both for single strings and lists, tuples or other types. It returns a string or list of strings as datetime object(s).

    Input: 
          string_date = str or list/tuple/etc. of strs, contains dates in string format

    Output:
           datetime_obj = datetime or list of datetimes, contains converted string(s) into datetime format as specified above. 
    '''
    
    if type(string_date) == str:
        datetime_obj = datetime.strptime(string_date, '%Y_%m_%d').date() 
    else:
        datetime_obj = []
        for i in range(len(string_date)):
            obj = datetime.strptime(string_date[i], '%Y_%m_%d').date() 
            datetime_obj.append(obj)
    return datetime_obj

def monthly_combinations(clean_df):
    '''
    STILL TO BE IMPROVED - not producing the correct output: combinations of input images if 5 images in a row,
    given the same month, respect the requirements of no-data and water thresholds. 

    Gets available datasets considering combinations for each month separately.

    Improvement to do: the function so far only works considering a time window of 4 years. It would need to be automated in order to
    use different time windows.
    
    Example: January 2001, January 2002, January 2003, January 2004 used as inputs to predict January 2005.
    '''
    # get all dates in the clean df
    dates_df = ['_'.join(clean_df['Date image'].iloc[i].split('_')[:3]) for i in range(len(clean_df))] # splits every '_' and then joins the first 3 (yyyy_mm_dd)

    years = [i for i in range(1987, 2022)]
    months = [i for i in range(1, 13)]

    dates = []

    for year in years:
        for month in months:
            if month < 10:
                date = f'{year}_0{month}_01'
            else: 
                date = f'{year}_{month}_01'
            dates.append(date)
    dates = dates[11:]

    datasets = []
    missing = []

    for element in dates:
        year = str((dates[dates == element].split('_'))[0])
        month = str((dates[dates == element].split('_'))[1])
        
        condition1 = f'{int(year)+1}_{month}_01' in dates_df
        condition2 = f'{int(year)+2}_{month}_01' in dates_df
        condition3 = f'{int(year)+3}_{month}_01' in dates_df
        condition4 = f'{int(year)+4}_{month}_01' in dates_df

        if condition1 and condition2 and condition3 and condition4:
            dataset = [f'{year}_{month}_01', f'{int(year)+1}_{month}_01', f'{int(year)+2}_{month}_01', f'{int(year)+3}_{month}_01', f'{int(year)+4}_{month}_01']
            datasets.append(dataset)
        else:
            miss = f'Starting {f"{year}_{month}_01"} misses: '
            if not condition1:
                miss += f'{int(year)+1}_{month}_01, '
            if not condition2:
                miss += f'{int(year)+2}_{month}_01, '
            if not condition3:
                miss += f'{int(year)+3}_{month}_01, '
            if not condition4:
                miss += f'{int(year)+4}_{month}_01'
            missing.append(miss)
            
    return datasets, missing

def get_month(df, single_month):
    '''
    This function filters the input dataframe based on the provided month. All images representative of all other months will be filtered out.
    This function is used in order to copy the images in a new folder before the dataset generation.

    Inputs:
           df = pandas DataFrame, contains monthly images with info on amount of pixels of different classes.
           month = int, specifies which month is used for the inputs/target combinations of the model

    Output:
           df_filtered = pandas DataFrame, original dataframe after filtering the images of the specified month 
    '''
    df_filtered = df[df['Date image'].apply(lambda x: x.month) == single_month]
    # df_filtered = pd.DataFrame(filtered_list) 
    return df_filtered

def assign_group(df, monsoon_start=5, monsoon_end=10):
    '''
    This function is used to assign a unique group to images based on their date.
    Images from November of year 1 until April of year 2 area ssigned to group 1. Then, images from May to October of year 2 are assigned to group 'Monsoon 1'.
    This procedure is repeated across the full dataframe.
    
    It is possible to set the months included in the monsoon season, where most of the images are likely to be cloudy (no-data) and therefore not usable.
    It is also relevant to point out that the images look different in May and October/November as well, depending on the year, because of the rising and falling flood stage,
    respectively. This causes the images to have more `water` pixels and therefore more submerged areas, with less morphological features (bars, secondary channels, etc.) visible.

    This is needed in order to group images at a later stage depending.

    Example: images from November 1987 to April 1988 --> group 1.
    Example: images from November 1987 to April 1988 --> group 1.
             images from May 1988 to October 1988 --> group 'Monsoon 1'.
             images from November 1988 to April 1989 --> group 2.
             and so on
    
    ATTENTION: the function needs improvement, as for the moment the group values are not in increasing in the expected order (1, 2, 3, etc.) but skipping some values.
    Despite this, each group is assigned with a unique identifier, therefore it is still acceptable for the goal it is required to.

    Input: 
          df = pandas DataFrame, contains monthly images with info on amount of pixels of different classes.
          monsoon_start, monsoon_end = int, month of the year representing the beginning and ending of monsoon season.
                                       default: 5 and 10, respectively. All months can be chosen but generally monsoon season
                                       goes from early June to September, although images are already quite cloudy between 
                                       May-October.

    Output: 
           df = pandas DataFrame, updated df with the new column 'Group' which contains the unique identifier to group images at a later stage
    '''
    if monsoon_end != 12:
        start_low_flow = monsoon_end + 1
    elif monsoon_end == 12:
        start_low_flow = 1 # set January as beginning of low flow conditions
    end_low_flow = monsoon_start - 1
    # initialize column
    df['Group'] = 0
    previous_group = 0

    for i in range(len(df)):
        # get image year
        year = df['Date image'].iloc[i].year 
        
        # apply group 'Monsoon i' to images between May and October (included)
        if df['Date image'].iloc[i].month >= monsoon_start and df['Date image'].iloc[i].month <= monsoon_end:
            df['Group'].iloc[i] = f"Monsoon {year - df['Date image'].iloc[0].year}"
        
        # apply group i to images between monsoon_end month of year i 
        # apply group i to images between monsoon_end month of year i 
        elif (df['Date image'].iloc[i].month >= start_low_flow and df['Date image'].iloc[i].year == year):
            previous_group += 1 
            
            if df['Date image'].iloc[i].month == start_low_flow: # sets group value based on previous_group update
                df['Group'].iloc[i] = previous_group // 2 
            
            else: # makes sure that group value is costant across all months of the same period
                df['Group'].iloc[i] = df['Group'].iloc[i-1] 
        
        # as above, but this is done for images between January and April of year i + 1
        elif (df['Date image'].iloc[i].month <= end_low_flow and df['Date image'].iloc[i].year == year):
            df['Group'].iloc[i] = df['Group'].iloc[i-1] 
    
    return df

def load_df_countpixels(train_val_test, reach_id, monsoon_start=None, monsoon_end=None, single_month=None, dir_folders = r'data\satellite\preprocessed', 
                        collection = r'JRC_GSW1_4_MonthlyHistory', cols = ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2']):
    '''
    This function loads the .csv file of a given use and reach. It creates a dataframe with the info of the .csv file (date and tot pixels per class)
    and creates three new columns ('Use', 'Reach' and 'Group'). 
    It uses the function `convert_to_date` to convert the image name to a datetime.date object and the function `assign_group` to assign a unique group to images, if the best image per season is chosen for the model training. 
    If the model is trained on all images of a given month instead, it makes use of the function `get_month`, which filters the images based on the specified month.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach_id = int, representing reach number. Number increases going upstream.
                      default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           monsoon_start, monsoon_end = int, month of the year representing the beginning and ending of monsoon season.
                                        default: None, both. In case the function is used for getting the best image among one monsoon season, both values need to be specified.
                                        All months can be chosen but generally monsoon season goes from early June to September, although images are already quite cloudy between 
                                        May-October. The suggested combination is `monsoon_start=5` and `mosnoon_end=10`.
           single_month = int, specifies which month is used for the inputs/target combinations of the model
                          default: None. In case the function is used to get all images of a specific month, a value needs to be set. All months can be chosen but it is recommended
                                   to use images from January to April to avoid those taken during the monsoon season and those taken just after, during the falling stage of the floods.
                                   This is suggested to avoid images with too many water pixels to be used during for the model training, as these are deemed to bring too much confusion
                                   to the model and also don't allow a clear recognition of morphological features. 
                                   Therefore, the suggested values are between `single_month=1` and `single_month=4`. 
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\preprocessed'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           cols = list of str, contains the name of columns to be loaded 
                  default: ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2']

    Output: 
           df = pandas DataFrame, contains total pixels per class, date of the image as well as use and reach id.
    '''
    
    full_path = os.path.join(dir_folders, collection + fr'_{train_val_test}_r{reach_id}', fr'{train_val_test}_r{reach_id}_allpixels.csv')
    df = pd.read_csv(full_path, usecols = cols) # skips additional index column present in the .csv files
    
    # convert string to datetime.date object
    dates = convert_to_date([df['Date image'].iloc[i] .split(f'_{train_val_test}', 1)[0] for i in range(len(df))]) # split name in two and take only first part 
    
    df['Date image'] = dates
    df['Use'] = train_val_test
    df['Reach'] = reach_id

    
    if monsoon_start != None and monsoon_end != None:
        # assign unique identifier to images within the same period
        df = assign_group(df, monsoon_start, monsoon_end) 
        # reorganize columns
        df = df.reindex(['Date image', 'Use', 'Reach', 'no-data: 0', 'non-water: 1', 'water: 2', 'Group'], axis=1)
    
    elif single_month != None:
        # filter dataframe considering only the specified month
        df = get_month(df, single_month)
        # reorganize columns
        df = df.reindex(['Date image', 'Use', 'Reach', 'no-data: 0', 'non-water: 1', 'water: 2'], axis=1)
    
    return df

def best_by_group(df, col_group = 'Group', col_min = 'no-data: 0'):
    '''
    This functions groups images based on the value of the assigned column `col_group`. 
    For each group It returns the image with the minimum amount of pixels of `col_min`.

    Inputs: 
           df = pandas DataFrame, contains info on each image given use and reach
           col_group = str, specifies column over used for grouping the images.
                       default: 'Group'
           col_min = str, specifies class from which access the image with minimum value
                     default: 'no-data: 0'
    
    Output: 
           best_img = pandas df, contains the best image (i.e., the one with min value of pixel of specified class) for each group
    '''
    # group images by assigned column 
    grouped_df = df.groupby(col_group)

    # get index of minimum of assigned column
    min_by_group = grouped_df[col_min].idxmin()

    # get relative rows
    best_img = df.loc[min_by_group]
    # sort by date 
    sorted_best = best_img.sort_values(by='Date image')
    return sorted_best

def imgs_for_dataset(train_val_test, reach_id, monsoon_start=None, monsoon_end=None, single_month=None, 
                     dir_folders = r'data\satellite\preprocessed', collection = r'JRC_GSW1_4_MonthlyHistory', 
                     cols = ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2'], col_group = 'Group', 
                     col_min = 'no-data: 0'):
    '''
    This functions loads .csv files containing info on amount of pixels per class and creates new columns. If `monsoon_start` and `monsoon_end` are specified,
    selects the images with the minimum pixels of a specified class per each group. Otherwise, if `single_month` is set, it selects the images representative of that month.
    It returns a dataframe containing these images, which will be used to create the dataset needed for training, validating and testing the model. 

    It is build upon the functions `load_df_countpixels` and `best_by_group`.

    Inputs: 
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach_id = int, representing reach number. Number increases going upstream.
                      default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           monsoon_start, monsoon_end = int, month of the year representing the beginning and ending of monsoon season.
                                        default: None, both. In case the function is used for getting the best image among one monsoon season, both values need to be specified.
                                        All months can be chosen but generally monsoon season goes from early June to September, although images are already quite cloudy between 
                                        May-October. The suggested combination is `monsoon_start=5` and `mosnoon_end=10`.
           single_month = int, specifies which month is used for the inputs/target combinations of the model
                          default: None. In case the function is used to get all images of a specific month, a value needs to be set. All months can be chosen but it is recommended
                                   to use images from January to April to avoid those taken during the monsoon season and those taken just after, during the falling stage of the floods.
                                   This is suggested to avoid images with too many water pixels to be used during for the model training, as these are deemed to bring too much confusion
                                   to the model and also don't allow a clear recognition of morphological features. 
                                   Therefore, the suggested values are between `single_month=1` and `single_month=4`. 
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\preprocessed'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           cols = list of str, contains the name of columns to be loaded 
                  default: ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2']
           col_group = str, specifies column over used for grouping the images.
                       default: 'Group'
           col_min = str, specifies class from which access the image with minimum value
                     default: 'no-data: 0'
    
    Output: 
           best_images = pandas DataFrame, contains best images (those with min no-data pixels or other specified column) per normal flow season
    '''
    # load dataframe
    df = load_df_countpixels(train_val_test, reach_id, monsoon_start, monsoon_end, single_month, dir_folders, collection, cols)
    
    if monsoon_start is not None and monsoon_end is not None:
        # get best images from each group
        best_images = best_by_group(df, col_group, col_min)
    elif single_month is not None:
        best_images = df

    return best_images 

def copy_images(train_val_test, reach_id,  monsoon_start=None, monsoon_end=None, single_month=None, dir_folders = r'data\satellite\preprocessed', 
                dir_dest = r'data\satellite\dataset', collection = r'JRC_GSW1_4_MonthlyHistory', 
                cols = ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2'], col_group = 'Group', col_min = 'no-data: 0'):
    '''
    This function copies the best images of each reach considering the non-monsoon season or the specific month (depending on the arguments specified, 
    either both `monsoon_start` and `monsoon_end` or `single_month`) in a new folder for starting the input dataset creation for the deep-learning model.

    It is built upon the function `imgs_for_dataset` that creates a dataframe containing the most suitable images for every season or from a specific month that
    will be used for the training, validation and testing of the model.

    Inputs:
           train_val_test = str, specifies for what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach_id = int, representing reach number. Number increases going upstream.
                      default: 1, applies for both validation and testing.
                   For training, the available range is 1-28 (included)
           monsoon_start, monsoon_end = int, month of the year representing the beginning and ending of monsoon season.
                                        default: None, both. In case the function is used for getting the best image among one monsoon season, both values need to be specified.
                                        All months can be chosen but generally monsoon season goes from early June to September, although images are already quite cloudy between 
                                        May-October. The suggested combination is `monsoon_start=5` and `mosnoon_end=10`.
           single_month = int, specifies which month is used for the inputs/target combinations of the model
                          default: None. In case the function is used to get all images of a specific month, a value needs to be set. All months can be chosen but it is recommended
                                   to use images from January to April to avoid those taken during the monsoon season and those taken just after, during the falling stage of the floods.
                                   This is suggested to avoid images with too many water pixels to be used during for the model training, as these are deemed to bring too much confusion
                                   to the model and also don't allow a clear recognition of morphological features. 
                                   Therefore, the suggested values are between `single_month=1` and `single_month=4`. 
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\preprocessed'
           dir_dest = str, destination directory where best images are stored.
                      default: r'data\satellite\dataset'  
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           cols = list of str, contains the name of columns to be loaded 
                  default: ['Date image', 'no-data: 0', 'non-water: 1', 'water: 2']
           col_group = str, specifies column over used for grouping the images.
                       default: 'Group'
           col_min = str, specifies class from which access the image with minimum value
                     default: 'no-data: 0'

    Output: 
           none, copies best images from the specified directory to the new assigned one for the dataset creation.
    '''
    # get full path
    dir_source = os.path.join(dir_folders, collection + fr'_{train_val_test}_r{reach_id}')
    dir_destination = os.path.join(dir_dest, collection + fr'_{train_val_test}_r{reach_id}')

    # create folders if not existing
    if not os.path.exists(dir_destination):
        os.makedirs(dir_destination)

    # retrieve best images
    best_images = imgs_for_dataset(train_val_test, reach_id, monsoon_start, monsoon_end, single_month, dir_folders, collection, cols, col_group, col_min)

    # copy imagse in the new folder
    for i in range(len(best_images)):
        if monsoon_start is not None and monsoon_end is not None:
            group = best_images[col_group].iloc[i] # group by specified group column
            if isinstance(group, int): # consider only rows whose specified group column is an integer (skips 'Monsoon i' images)
                year, month, day = str(best_images['Date image'].iloc[i]).split('-') # get date information
                file_path = os.path.join(dir_source, fr"{year}_{month}_{day}_{train_val_test}_r{reach_id}.tif")
                dest_path = os.path.join(dir_destination, fr"{year}_{month}_{day}_{train_val_test}_r{reach_id}.tif")
                shutil.copy(file_path, dest_path)
        elif single_month is not None:
            year, month, day = str(best_images['Date image'].iloc[i]).split('-') # get date information
            file_path = os.path.join(dir_source, fr"{year}_{month}_{day}_{train_val_test}_r{reach_id}.tif")
            dest_path = os.path.join(dir_destination, fr"{year}_{month}_{day}_{train_val_test}_r{reach_id}.tif")
            shutil.copy(file_path, dest_path)
    return None