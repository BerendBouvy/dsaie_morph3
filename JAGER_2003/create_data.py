import imageFeaturesClass as ifc
import os
import pandas as pd
import matplotlib.pyplot as plt


def undersample_data(file, percentage=1):
    df = pd.read_csv(file)
    features = ['distance', 'sin_angle', 'cos_angle', 'river_width', 'water_in_range', 'next_year_water']

    target1 = df[df[features[-1]] == 1]
    target0 = df[df[features[-1]] == 0]
    target0 = target0.sample(int(percentage*len(target1)), random_state=47)
    df = pd.concat([target0, target1], ignore_index=True)
    new_path = file.split('/')
    new_path[-1] = 'undersampled_' + new_path[-1] 
    new_path = '/'.join(new_path)
    df.to_csv(new_path, index=False)

def merge_files(path):
    files = os.listdir(path)
    feature_files = [file for file in files if 'features' in file]
    dfs = [pd.read_csv(f"{path}/{file}") for file in feature_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(f"{path}/merged_features.csv", index=False)
    print("merged_features.csv created")
    undersample_data(f"{path}/merged_features.csv")


def create_data(path):
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
    # ask input folder
    # path = input("Please enter the location indicator: ")
    # path = "data/satellite/averages/average_training_" + path
    path = "data/satellite/averages/average_testing_r1"
    create_data(path)   
    # lst = os.listdir("data/satellite/averages")
    # lst = lst[2:]
    # for path in lst:
    #     create_data(f"data/satellite/averages/{path}")
    # merge_files("data/satellite/averages/average_testing_r1")
    undersample_data("data/satellite/averages/average_testing_r1/merged_features.csv", 3)
