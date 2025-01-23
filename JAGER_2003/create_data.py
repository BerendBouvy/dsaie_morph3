import imageFeaturesClass as ifc
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_data(path):
    files = os.listdir(path)
    
    for file in files:
        image = ifc.ImageFeatures(f"{path}/{file}", 7)
        df = image.get_features()
        df.to_csv(f"{path}/features_{file}")
        print(f"features_{file} created")
        

def merge_files(path):
    files = os.listdir(path)
    feature_files = [file for file in files if 'features' in file]
    dfs = [pd.read_csv(f"{path}/{file}") for file in feature_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(f"{path}/merged_features.csv", index=False)
    print("merged_features.csv created")

def undersample_data(file):
    df = pd.read_csv(file)
    features = ['distance', 'sin_angle', 'cos_angle', 'river_width', 'water_in_range', 'next_year_water']

    target1 = df[df[features[-1]] == 1]
    target0 = df[df[features[-1]] == 0]
    target0 = target0.sample(target1.shape[0], random_state=47)
    df = pd.concat([target0, target1], ignore_index=True)
    new_path = file.split('/')
    new_path[-1] = 'undersampled_' + new_path[-1] 
    new_path = '/'.join(new_path)
    df.to_csv(new_path, index=False)



if __name__ == "__main__":
    # ask input folder
    # path = input("Please enter the location indicator: ")
    # path = "data/satellite/averages/average_training_" + path
    # # create_data(path)   
    # merge_files(path) 
    undersample_data('data/satellite/averages/average_training_r1/merged_features.csv')