"""
This script processes satellite image data to extract features, merge them, and perform undersampling.
Functions:
    undersample_data(file, percentage=1):
        Reads a CSV file, undersamples the data based on the target feature, and saves the undersampled data to a new CSV file.
    merge_files(path):
        Merges multiple feature CSV files in a given directory into a single CSV file and performs undersampling on the merged data.
    create_data(path):
        Processes each file in the given directory to extract image features, save them to CSV files, and merge them into a single file.
Usage:
    Run the script and provide the path to the directory containing the satellite image data files.
"""
import imageFeaturesClass as ifc
import os
import pandas as pd
import matplotlib.pyplot as plt


def undersample_data(file, percentage=1):
    """
    Reads a CSV file, undersamples the data based on the target feature, and saves the undersampled data to a new CSV file.
    Args:
        file (str): The path to the CSV file to be undersampled.
        percentage (float): The percentage of the majority class to be sampled. Default is 1.
    """
    # Read the CSV file
    df = pd.read_csv(file)
    # Define the features and target variable
    features = ['distance', 'sin_angle', 'cos_angle', 'river_width', 'water_in_range', 'next_year_water']
    # Undersample the data based on the target variable
    target1 = df[df[features[-1]] == 1]
    target0 = df[df[features[-1]] == 0]
    target0 = target0.sample(int(percentage*len(target1)), random_state=47)
    df = pd.concat([target0, target1], ignore_index=True)
    # Save the undersampled data to a new CSV file
    new_path = file.split('/')
    new_path[-1] = 'undersampled_' + new_path[-1] 
    new_path = '/'.join(new_path)
    df.to_csv(new_path, index=False)
    print(f"{new_path} created")

def merge_files(path):
    """
    Merges multiple feature CSV files in a given directory into a single CSV file and performs undersampling on the merged data.
    Args:
        path (str): The path to the directory containing the feature CSV files to be merged.
    """
    files = os.listdir(path)
    feature_files = [file for file in files if 'features' in file]
    dfs = [pd.read_csv(f"{path}/{file}") for file in feature_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(f"{path}/merged_features.csv", index=False)
    print("merged_features.csv created")
    undersample_data(f"{path}/merged_features.csv")


def create_data(path):
    """
    Processes each file in the given directory to extract image features, save them to CSV files, and merge them into a single file.
    Some files may be empty, in which case the previous file is also removed.
    Args:
        path (str): The path to the directory containing the satellite image data files.
    """        
    files = os.listdir(path)
    
    for file in files:
        if file != files[-1]:
            image = ifc.ImageFeatures(f"{path}/{file}", 7)
            df = image.get_features()
            if not df.empty:
                df.to_csv(f"{path}/features_{file}")
                print(f"features_{file} created")
            if df.empty:
                # remove previous file
                file_split = file.split('_')
                file_year = int(file_split[1])
                previous_file = f"features_average_{file_year-1}_{file_split[2]}_{file_split[3]}"
                if previous_file in os.listdir(path):
                    os.remove(f"{path}/{previous_file}")
                print(f"{file} was empty, {previous_file} removed")
    merge_files(path)
        

if __name__ == "__main__":
    """	Main function to run the script. Some examples are provided below. """
    # ask input folder
    # path = input("Please enter the location indicator: ")
    # path = "data/satellite/averages/average_training_" + path
    # path = "data/satellite/averages/average_testing_r1"
    # create_data(path)   
    # lst = os.listdir("data/satellite/averages")
    # lst = lst[2:]
    # for path in lst:
    #     create_data(f"data/satellite/averages/{path}")
    # merge_files("data/satellite/averages/average_testing_r1")
    # undersample_data("data/satellite/averages/average_testing_r1/merged_features.csv", 3)
    pass
