# JAGER_2003 ANN Model guide
This set of scripts can be used to train a model to predict erosion along a braided river. It is based on a neural network described in a paper of Jager, 2003.

To train this model properly it is recommended to run the script ANN_model_initiation.py from the commandline. Aside from this there is an example jupyter notebook which shows the process on how the model is trained and the functions used.

## Files
- imageFeaturesClass.py: Contains the functions needed to extract the features used for training the model from the dataset.
- create_data.py: Scripts that merges the features from multiple datasets to create a dataset for training, it undersamples the data to create a better balance between the different targets. By setting the path in this file, you can indicate where the data is stored and which data to process. It automatically undersamples the data, to get a better balance between different targets.
- ANN.py: Contains the functions needed to create the neural network and train it.
- ANN_model_initiation.py: Script that creates the neural network and trains it based on the parameters defined in the script. It can train multiple models, by listing multiple values for each parameter.
- train_models.py: Contains functions that loops over the different parameters defined in ANN_model_initation.py to create, train and store the different neural networks.
- Example_ANN_model_creation.ipynb: Notebook, which gives an example of how the neural network can be created and trained. It is not recommended to be used for model training, only to be used for reference.
- test.py: Contains functions used to test the imageFeaturesClass.py
- plot_pred.py: Script to plot the predictions and the metrics of the saved models.
- training_plotter.py: Function used to create loss plot and save this plot.

## Quick Start Guide
To train this model the following folder Layout is needed. The data folder contains the data from Antonio Magherini his thesis on JamUNet.

### 1. Folder structure

<div style="background-color: lightgrey;>

* [data](.\data)
    * [satellite](.\data\satellite)
        * [averages](.\data\satellite\averages)
* [JAGER_2003](.\JAGER_2003)
    * [test_r1_data/](.\JAGER_2003\test_r1_data)
        * [merged_features.csv](.\JAGER_2003\test_r1_data\merged_features.csv)
        * [undersampled_merged_features.csv](.\JAGER_2003\test_r1_data\undersampled_merged_features.csv)
* [images](.\images)
* [models](.\models)

</div>

### 2. Preprocessing the data
1. Open ./JAGER_2003/create_data.py to select the dataset to extract features from.
2. Run the following code in the terminal
    * <code> python JAGER_2003/create_data.py </code>
3. Place the data in a folder named ./JAGER_2003/test_location_data or ./JAGER_2003/training_location_data as indicated in the folder structure., with location as r1, r2, etc and choose between training or test. 

### 3. Training the model

1. Open the file ANN_model_initiation.py and configure the neural network you want to train. 
2. Save the file.
3. Move in terminal to repository in your directory
4. Run the following code in the terminal 
    * <code> python JAGER_2003/ANN_model_initiation.py</code>
5. Indicate which data you want to use, recommended: test_r1
6. Indicate which year to use as test data, recommended: 2020

### 4. Visualize the predictions

1. Open the file plot_pred.py and select the prediction location you want to visualize from step 3. These are found in the folder models.
2. Save the file.
3. Run the following code in the terminal
    * <code> python JAGER_2003/plot_pred.py </code>


