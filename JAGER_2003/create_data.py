import imageFeaturesClass as ifc
import os

def create_data(path):
    files = os.listdir(path)
    
    for file in files:
        image = ifc.ImageFeatures(f"{path}/{file}", 7)
        df = image.get_features()
        df.to_csv(f"{path}/features_{file}")
        print(f"features_{file} created")
        

    # write to csv




if __name__ == "__main__":
    path = "data/satellite/averages/average_training_r2"
    create_data(path)    