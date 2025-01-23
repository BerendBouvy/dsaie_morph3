import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

class ANN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=1, hidden_nodes=5, activation='relu'):
        """
        Create a simple feedforward neural network with the specified input and output dimensions, number of hidden layers,
        number of nodes in each hidden layer, and activation function.
        :param input_dim: int, number of input features
        :param output_dim: int, number of output features
        :param hidden_layers: int, number of hidden layers (default=1)
        :param hidden_nodes: int, number of nodes in each hidden layer (default=5)
        :param activation: str, activation function to use (default='relu')
        """
        super(ANN, self).__init__()

        # Initialize the input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the hidden layers and nodes
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes

        # Create the neural network

        if activation == 'relu':
            
            network = []
            network.append(nn.Linear(self.input_dim, self.hidden_nodes))
            network.append(nn.ReLU())
            for i in range(self.hidden_layers-1):
                network.append(nn.Linear(self.hidden_nodes, self.hidden_nodes))
                network.append(nn.ReLU())
            network.append(nn.Linear(self.hidden_nodes, self.output_dim))

        else:
            print(f"Activation function {activation} not supported. \nCheck docs for supported activation functions.")    

        self.network = nn.Sequential(*network)

    def forward(self, x):
        x = self.network(x)
        outputs = torch.sigmoid(x)

        return outputs
    
    def classify(self, x):
        y = self.forward(x).detach()
        y_class = torch.where(y < 0.5, torch.tensor(0), torch.tensor(1))
        
        return y_class, y

class normUnitvar:
    """
    Class to normalize and denormalize data using the mean and standard deviation of the full dataset.
    """
    def __init__(self, fullDataset):
        self.normmean = fullDataset.mean(axis=0)
        self.normstd = fullDataset.std(axis=0)

    def normalize(self, data):
        return (data - self.normmean) / self.normstd

    def denormalize(self, data):
        return data * self.normstd + self.normmean
    
def cross_entropy(y, t):
    """
    Compute the cross entropy loss between the predicted and target values.
    :param y: torch.Tensor, predicted values
    :param t: torch.Tensor, target values
    :return: torch.Tensor, cross entropy loss
    """
    c_e = -torch.sum(t * torch.log(y) + (1 - t) * torch.log(1 - y))
    
    return c_e

def optimParameters(
        model, params, train_loader, val_loader, lambda_val = 0.01, n_epochs = 2000, learning_rate = 0.01
    ):
    """
    Function to optimize the parameters of a neural network model using the Adam optimizer.
    :param model: nn.Module, neural network model
    :param params: dict, dictionary of optimization parameters
    :param train_loader: DataLoader, training data loader
    :param val_loader: DataLoader, validation data loader
    :param lambda_val: float, regularization parameter (default=0.01)
    :param n_epochs: int, number of training epochs (default=2000)
    :return best_model: nn.Module, neural network model with the best validation loss
    :return best_loss: float, best validation loss
    :return metrics: dict, dictionary of training and validation loss
    """
    # Initialize the optimizer
    adam = torch.optim.Adam(params, lr=learning_rate, weight_decay=lambda_val)
    best_loss = 1e10 # High enough to always be lowered by the first loss

        # Train the model
    for epoch in range(n_epochs):
        # Initialize Metrics
        metrics_temp = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0
        }
        # Training data
        for data in train_loader:
            inputs, targets = data
            outputs = model(inputs)
            loss = cross_entropy(outputs, targets)

            adam.zero_grad()
            loss.backward()
            adam.step()
        metrics_temp['train_loss'].append(loss)

        # Validation data
        val_loss = 0
        for data in val_loader:
            inputs, targets = data
            outputs = model(inputs)
            val_loss += cross_entropy(outputs, targets)
        metrics_temp['val_loss'].append(val_loss)

        # Compare current model with best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            metrics = metrics_temp

        # Check for improvements for 50 epochs
        if epoch > best_epoch + 50:
            break

        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")

    print(f"Final epoch: {epoch}, loss: {val_loss}, best model at epoch {best_epoch} with loss {best_loss}")

    # Store the best epoch in metrics        
    metrics['best_epoch'] = best_epoch

    return best_model, best_loss, metrics

def createDataLoader(dataset, batch_size=8, generator=False):
    """
    This Function splits the dataset in a training and validation dataset. And returns them in dataloaders.

    Parameters
    ----------
    dataset 
        Dataset used to create model.
    batch_size : int, optional
        Size of the batches created for the dataloader, by default 8
    generator : bool, optional
        Use a see for the random_split for testing purposes, by default None

    Returns
    -------
    train_loader : torch.utils.data.dataloader.DataLoader
        Dataloader containing the training dataset in batches
    val_loader : torch.utils.data.dataloader.Dataloader
        Dataloader containing the validation dataset in batches
    """
    # Split dataset into a training and validation set
    train_split = int(0.8 * len(dataset))
    val_split = len(dataset) - train_split

    # For testing Generator can be used to create a DataLoader else no generator
    if generator is True:
        generator_1 = torch.Generator().manual_seed(0)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_split, val_split], generator_1
        )
    else:
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_split, val_split]
        )
    
    #Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader