from matplotlib import pyplot as plt
import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import DataLoader, TensorDataset
gdal.UseExceptions()

def tiff_to_csv(tiff_path, plot=False):
    """
    Converts a TIFF file to a CSV file.
    Parameters:
    tiff_path (str): The file path to the TIFF file.
    plot (bool): If True, displays the TIFF image using matplotlib. Default is False.
    Raises:
    FileNotFoundError: If the TIFF file cannot be opened.
    Returns:
    None
    """
    
    # Open the TIFF file
    name = tiff_path.split('/')[-1].split('.')[0]
    csv_path = f"{name}.csv"
    dataset = gdal.Open(tiff_path)
    
    if dataset is None:
        raise FileNotFoundError(f"Unable to open {tiff_path}")

    # Read the raster data as a numpy array
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    # Save the array to a CSV file
    plt.imshow(array)
    plt.show() if plot else None
    np.savetxt(csv_path, array, delimiter=",", fmt='%d')
    
def pt_to_csv(pt_path, plot=False):
    """
    Converts a PyTorch tensor saved in a file to a CSV file.
    Args:
        plot (bool, optional): If True, plots the tensor as an image. Default is False.
    Raises:
        ValueError: If the file does not contain a valid PyTorch tensor.
    Returns:
        None
    """
    # Load the tensor from the .pt file
    tensor = torch.load(pt_path, map_location=
                        torch.device('cpu'))

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"The file {pt_path} does not contain a valid PyTorch tensor")
    
    # Convert the tensor to a numpy array
    array = tensor.numpy()
    
    # Save the array to a CSV file
    name = pt_path.split('/')[-1].split('.')[0]
    folder = '/'.join(pt_path.split('/')[:-1])
    csv_path = f"{folder}/{name}.csv"
    np.savetxt(csv_path, array, delimiter=",", fmt='%d')
    
    # Optionally plot the array
    if plot:
        plt.imshow(array)
        plt.show()

# Example usage
if __name__ == "__main__":
    # tiff_to_csv("data/satellite/dataset_month3/JRC_GSW1_4_MonthlyHistory_testing_r1/2020_03_01_testing_r1.tif", plot=True)
    pt_to_csv("data/test_set_month3.pt")