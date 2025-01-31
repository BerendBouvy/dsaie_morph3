# This module contains the functions used for preprocessing the river data  
# (discharge, water level, velocity) needed for the data analysis  

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# specify path of flow data 
path_flow_data = r'data/flow/original'

# set formats for dataframes data type
format_main = '%d/%m/%Y'
format_alternative = '%d %B, %Y %I:%M:%p'
format_alternative2 = '%Y-%m'

# set formats for plotting 
format_day = '%b'
format_year = '%Y'

def load_Q_baha_94_00(file, header, cols, name_cols, date_format):
    '''
    Functions used for creating a pandas dataframe from the file 'Daily_Q_Bahadurabad_1994-2000.xlsx' which contains daily records of the discharge at Bahadurabad between 1994-2000.

    Inputs:
           file = str, specifies file name
           header = int, specifies number of row to use as header for the pandas dataframe
           cols = int or list of ints, specifies columns to load
           name_cols = str or list of strs, names for the dataframe columns
           date_format = str, specifies date format. 
    
    Outputs: 
            df = pandas df, contains the recorded data
    '''
    df = pd.read_excel(path_flow_data + file, header=header, usecols=cols)
    # rename columns
    df.columns = name_cols
    df[name_cols[0]] = pd.to_datetime(df[name_cols[0]], format=date_format)
    return df

def load_baha_69_94(file_path):
    '''
    Functions used for creating a pandas dataframe from the file 'Daily_WL_Bahadurabad_1964-1994_test.xlsx' which contains daily records of the water level at Bahadurabad between 1964-1994.

    Inputs: 
           file_path = str, contains path of the excel file with recorded data

    Outputs:  
            df = pandas dataframe, contains the recorded data
    '''

    df = pd.read_excel(path_flow_data + file_path, header=4) 
    # drop unnecessary rows and columns
    df.drop(axis=0, index=[i for i in range(367, len(df))], inplace=True)
    df.drop(axis=0, index=0, inplace=True)
    df.drop(axis='columns', columns=df.columns[32:], inplace=True)
    # rename columns
    years = [i for i in range(1964, 1995)]
    cols_names = ['Date (dd-mm)']
    cols_names.extend(str(item) for item in years) # years
    df.columns = cols_names
    # change datatype to date (only dd-mm, year not included)
    df['Date (dd-mm)'] = pd.to_datetime(df['Date (dd-mm)']).dt.strftime('%d-%m')
    # remove 29/02 from records - only 6 values in 30 years with no relevant deviation from values at that time of the year
    df = df[df['Date (dd-mm)'] != '29-02'] 
    df.reset_index(drop=True, inplace=True)
 
    return df

def load_baha_94_00(file, header, cols, name_cols, date_format):
    ''''
    Functions used for creating a pandas dataframe from the file 'Monthly_Q_Bahadurabad_1994-2000.xlsx' 
    which contains monthly records (min, avg and max) of the discharge at Bahadurabad between 1994-2000.

    Inputs:
           file = str, specifies file name
           header = int, specifies number of row to use as header for the pandas dataframe
           cols = int or list of ints, specifies columns to load
           name_cols = str or list of strs, names for the dataframe columns
           date_format = str, specifies date format. 
    
    Outputs: 
            df = pandas df, contains the recorded data
    '''
    df = pd.read_excel(path_flow_data + file, header=header, usecols=cols)
    df['Date (yyyy-mm-dd)'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str), format=date_format) # create dataframe column - day automatically set to 1
    df.drop(columns=['YEAR', 'MONTH'], inplace=True) # remove initial column 
    df.columns = name_cols
    df = df[[df.columns[-1]] + df.columns[:-1].tolist()] # reorder columns - date first
    return df

def fill_na(df, axis, fill_method='mean'):
    '''
    This function is used for filling NaN values with a chosen method
    
    Inputs: 
           df = pandas dataframe, contains recorded data
           rows, cols = int or slices, specify rows and columns over which 
                        the function is applied
           axis = int, binary choice between 0 and 1 (row or column) over which 
                  the method is used to fill NaN
           fill_method = str, specifies which method is used for filling NaN values
                         default: 'mean' 
    
    Outputs:
            df = pandas dataframe with filled NaN values
    '''
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(
        lambda row: row.fillna(row.mean()), axis=axis).round(2)
    
    return df

def reshape_df(initial_df, time_array, format):
    '''
    Function used to reshape WL_d_Baha_64_94 dataframe which has one column per year of record.
    It returns a one-column output with years stacked vertically.

    Inputs:
           initial_df = pandas DataFrame, contains dates and measured variables 
           time_array = array, contains date from starting to ending date of the original dataframe with  daily interval
           format = str, specifies date format

    Output:
           df_reshaped = pandas DataFrame, reshaped original dataframe
    '''
    df_reshaped = pd.DataFrame(pd.to_datetime(time_array, format=format))
    df_reshaped.rename(columns={df_reshaped.columns[0]: 'Date (yyyy-mm-dd)'}, inplace=True)

    # initialize list with all recorded water levels between 1964 and 1994
    list_wl = []

    for year in (initial_df.columns[1:]):
            for day in range(len(initial_df['Date (dd-mm)'])):
                    list_wl.append(initial_df[year].iloc[day])

    df_reshaped['Average daily water level (m)'] = list_wl
    return df_reshaped

def fill_missing_data(df, variable, freq='D'):
    '''
    Function used for completing the dataframes when specific dates were not recorded. 
    The missing data are replaced with the average of the day/month across the recorded years.

    Inputs: 
           df = pandas DataFrame, contains original recorded data with missing values
           variable = str, specifies the recorded variable
                      available options: 'Average daily discharge ($m^3/s$)' 
                                         'Average daily water level (m)'
           freq = str, specifies time interval between two consecutive elements of time_interval array 
                  and determines the mean frequency
                  default: 'D', day 
                  other option available: 'MS', month start (01-xx)
    Output:
           df_filled = pandas DataFrame, contains full time interval with data replaced when missing 
    '''
    # get initial and end date and create time array
    start_date = df['Date (yyyy-mm-dd)'].min()
    end_date = df['Date (yyyy-mm-dd)'].max()
    time_interval = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # create new dataframe
    dates_df = pd.DataFrame(time_interval, columns=['Date (yyyy-mm-dd)']) 
    
    # merge original and new dataframe by keeping the recorded data for matching dates
    # missing dates will contain NaN values 
    df_filled = pd.merge(dates_df, df, on='Date (yyyy-mm-dd)', how='left')

    # group data by day/month and compute average across years
    if freq == 'D':
        mean_by_dt = df_filled.groupby(df_filled['Date (yyyy-mm-dd)'].dt.dayofyear).mean()
    elif freq== 'MS':  
        mean_by_dt = df_filled.groupby(df_filled['Date (yyyy-mm-dd)'].dt.month).mean()
    else:
        raise ValueError(f'You choose a {freq} but this is not available. The available options are "D" and "MS"')

    # replace NaN values with average across years 
    for index, row in df_filled.iterrows():
        if pd.isna(row[variable]):  # get NaN rows
            if freq == 'D':
                date = row['Date (yyyy-mm-dd)'].dayofyear
            elif freq=='MS':
                date = row['Date (yyyy-mm-dd)'].month
            mean_value = mean_by_dt.loc[date][variable]  # get mean of the specific day
            df_filled.at[index, variable] = mean_value  # replace value
    
    return df_filled

# --------------- #
# the next part of the module contains intitated functions but never used so far 

def get_info(df):
    '''
    This function is still not implemented. It will be done as soon as possible or never.

    Function used to get info on data regarding measured variable, measurement frequency, location,
    and years of record.

    Input:
          df = pandas DataFrame, contains dates and measured variables 
    '''
    # specify measured variable
    # if 'WL' is in str(name of df):
    #     var1 = 'water level'
    # elif 'Q':
    #     var1 = 'discahrge'
    # elif 'Umax':
    #     var1 = 'max velocity'

    # specify data measurement frequency
    # if '_d_' is in str(name of df):
    #     time_int = 'daily'
    # elif '_m_':
    #     time_int = 'monthly'
    # elif '_bw_':
    #     time_int = 'biweekly'
    
    # specify data measurement location
    # if '_Baha_' is in str(name of df):
    #     loc = 'Bahadurabad'
    # elif '_Sir_':
    #     loc = 'Sirjganj'
    # elif '_Ari_':
    #     loc = 'Aricha'

    # retrieve recorded years
        
    return None # var, time_int, loc, rec_yrs  