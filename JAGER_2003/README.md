# JAGER_2003 ANN Model guide
This set of scripts can be used to train a model to predict erosion along a braided river. It is based on a neural network described in a paper of Jager, 2003.


## Quick Start Guide
To train this model the following folder Layout is needed. The data folder contains the data from Antonio Magherini his thesis on JamUNet.

### 1. Folder structure
* [data](.\data)
    * [satellite](.\data\satellite)
        * [averages](.\data\satellite\averages)
* [JAGER_2003](.\JAGER_2003)
    * [test_r1_data/](.\JAGER_2003\test_r1_data)
        * [merged_features.csv](.\JAGER_2003\test_r1_data\merged_features.csv)
        * [undersampled_merged_features.csv](.\JAGER_2003\test_r1_data\undersampled_merged_features.csv)
* [images](.\images)
* [models](.\models)

### 2. Preprocessing the data

Place the data in a folder named training/test_location_data as indicated in the folder structure., with location as r1, r2, etc and choose between training or test. 

### 3. Training the model

1. Open the file ANN_model_initiation.py and configure the neural network you want to train. 
2. Save the file.
3. Move in terminal to repository in your directory
4. Run the following code in the terminal 
    * <code> python JAGER_2003/ANN_model_initiation.py</code>
5. Indicate which data you want to use, recommended: test_r1
6. Indicate which year to use as test data, recommended: 2020

### 4. Visualize the predictions


## To do:
- Write section 1 and 4 of readme. 

