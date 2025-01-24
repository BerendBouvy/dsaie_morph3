from train_models import train_models

# This script initiates the parameters used to train the ANN model
# The model is trained with different hyperparameters and the best models are saved
# It goes through all combinations of lambda, hidden_layers and hidden_nodes
if __name__ == "__main__":
    # Ask input folder
    # Due to skewed dataset, the undersampled dataset will be used for training,
    # The full dataset is used to extract the test data
    path_loc = input("Please enter the location indicator as test_r? or training_r?: ")
    path = f"JAGER_2003/{path_loc}_data/merged_features.csv"
    path2 = f"JAGER_2003/{path_loc}_data/undersampled_merged_features.csv"

    # Set model parameters
    model_params = {
        'lambda': [0],   # Array of regularization parameters
        'input_dim': 5,       # Number of input features
        'output_dim': 1,      # Number of outputs
        'hidden_layers': [10, 20],   # Array of number of hidden layers to be tested
        'hidden_nodes': [5, 10, 20],    # Array of number of nodes in hidden layers to be tested
        'activation': 'relu',   # Activation function for neurel network (relu, sigmoid)
        'learning_rate': 0.001,   # Learning rate for the optimizer
        'generator': True    # Use generator with seed(0) for data loader (True/False)
    }

    # Select the year to be used as test data
    test_year = input("Please enter the test year: ") 
    test_year = int(test_year)

    train_models(path, path2, model_params, test_year, path_loc)