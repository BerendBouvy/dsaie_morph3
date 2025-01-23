import numpy as np
from ANN import ANN, optimParameters, createDataLoader, normUnitvar, cross_entropy
import copy
import torch
import pandas as pd
import pickle
from training_plotter import plot_training

def train_models(path, path2, model_params, test_year):
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Parameters used for model training
    lambda_values = model_params['lambda']
    input_dim = model_params['input_dim']
    output_dim = model_params['output_dim']
    hidden_layers = model_params['hidden_layers']
    hidden_nodes = model_params['hidden_nodes']
    activation_function = model_params['activation']
    learning_rate = model_params['learning_rate']

    # Load dataset
    data = pd.read_csv(path2)
    data_test = pd.read_csv(path)
    features = ['distance', 'sin_angle', 'cos_angle', 'river_width', 'water_in_range'] 
    target_variable = ['next_year_water']
    data_test = data_test[data_test['year'] == test_year]
    data_training = data[data['year'] != test_year]

    # Normalize the features in the dataset per feature
    X = torch.tensor(data_training[features].values, dtype=torch.float32).to(device)
    X_normalizer = normUnitvar(X)

    X_test = torch.tensor(data_test[features].values, dtype=torch.float32).to(device)
    X_test_norm = X_normalizer.normalize(X_test)

    # Store the Targets in a tensor
    targets = torch.tensor(data_training[target_variable].values, dtype=torch.int8).to(device)
    T_test = torch.tensor(data_test[target_variable].values, dtype=torch.int8).to(device)

    # Create a new dataset with the normalized features
    X_norm = X_normalizer.normalize(X)

    # Create a new dataset with the normalized features and the targets
    train_loader, val_loader = createDataLoader(torch.utils.data.TensorDataset(X_norm, targets), batch_size=256)

    min_loss = 1e9
    best_lambda = 0

    # Loop over different neural network architectures and 
    # determine the best regularization parameter
    for h_layers in hidden_layers:
        for h_nodes in hidden_nodes:
            for lambda_val in lambda_values:
                #Initialize model
                model = ANN(input_dim, output_dim, h_layers, h_nodes, activation_function).to(device)
                w = model.parameters()

                # Train model
                model, val_loss, metrics = optimParameters(model, w, train_loader, val_loader, lambda_val, n_epochs=2000, learning_rate=learning_rate, device=device)

                print(f"Model with lambda = {lambda_val} has a loss of {val_loss}\n")

                #Save the best model
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_lambda = lambda_val
                    best_model = copy.deepcopy(model)
                    best_metrics = metrics
                    

            print(f"For {h_layers} hidden layers and {h_nodes} nodes,")
            print(f"the best lambda={best_lambda} with a loss of {min_loss}.\n")

            # Determine the test loss
            t_hat = best_model(X_test_norm)
            test_loss = cross_entropy(t_hat, T_test)
            test_loss = test_loss.cpu().detach().numpy()
            print(f"Test loss for the best model is: {test_loss}")

            # Save the predictions vs the actual values
            df_predictions = pd.DataFrame({
                'Targets': T_test.cpu().detach().numpy().tolist(), 
                'Predictions': t_hat.cpu().detach().numpy().tolist(),
                'index_x': data_test['index_x'].values,
                'index_y': data_test['index_y'].values
                })
            
            df_predictions.to_csv(f"models/predictions_{h_layers}_{h_nodes}_{best_lambda}.csv")

            # Save the best model and the metrics
            torch.save(best_model, f"models/best_model_{h_layers}_{h_nodes}_{best_lambda}.pth")
            np.save(f"models/metrics_{h_layers}_{h_nodes}_{best_lambda}.npy", best_metrics)

            # Save validation loss and training loss graph
            plot_training(best_metrics, h_layers, h_nodes, best_lambda, test_loss)

    # Save the training normalizer
    with open("models/data_normalizer_class.pkl", 'wb') as output:
        pickle.dump(X_normalizer, output, pickle.HIGHEST_PROTOCOL)    

if __name__ == "__main__":
    # Ask input folder
    # Due to skewed dataset, the undersampled dataset will be used for training,
    # The full dataset is used to extract the test data
    path_loc = input("Please enter the location indicator as test_r? or training_r?: ")
    path = f"JAGER_2003/{path_loc}_data/merged_features.csv"
    path2 = f"JAGER_2003/{path_loc}_data/undersampled_merged_features.csv"
    
    # Set model parameters
    model_params = {
        'lambda': [0, 0.001, 0.01],   # Array of regularization parameters
        'input_dim': 5,       # Number of input features
        'output_dim': 1,      # Number of outputs
        'hidden_layers': [1, 10, 25],   # Array of number of hidden layers to be tested
        'hidden_nodes': [5, 20, 50],    # Array of number of nodes in hidden layers to be tested
        'activation': 'relu',   # Activation function for neurel network
        'learning_rate': 0.01   # Learning rate for the optimizer
    }

    # Select the year to be used as test data
    test_year = input("Please enter the test year: ") 

    train_models(path, path2, model_params, test_year)