import imageFeaturesClass as ifc
import os
import pandas as pd

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



if __name__ == "__main__":
    path = "data/satellite/averages/average_training_r2"
    create_data(path)    