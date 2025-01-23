import matplotlib.pyplot as plt

def plot_training(metrics, hidden_layers, hidden_nodes, lambda_value, test_loss):
    """Function to create a plot of the training and validation loss for a given model and store the plot.

    Parameters
    ----------
    metrics : dict
        Dictionary containing the training and validation loss per epoch and the best epoch
    hidden_layers : int
        Number of hidden layers used in the model
    hidden_nodes : int
        Number of nodes per hidden layer used in the model
    lambda_value : float
        Regularization parameter used to optimize the model
    test_loss : float
        Loss of the model on the test set
    """

    epochs = np.arrange(0,len(metrics['train_loss']))
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']

    # Plot the training and validation loss and save the image
    plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='green')
    plt.scatter(metrics['best_epoch'], test_loss, label='Test Loss at best epoch', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for lambda={lambda_value}')
    plt.legend()
    plt.savefig(f'images/loss_plot_{hidden_layers}_{hidden_nodes}.png')
