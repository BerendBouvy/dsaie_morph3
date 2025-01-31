# This module contains the functions used for saving the images for the report

import os 

import matplotlib.pyplot as plt

from preprocessing.images_analysis import show_image_array  

def save_image_figure(fig, folder=None, output_name=None, output_path=r'images\report', dpi=600, format='png',
                      cmap='gray', vmin=0, vmax=2): 
    '''
    Save the figure to the specified path. If `folder` and/or `output_name` are not specified it raises an Exception. 
    
    Inputs:
           fig = matplotlib.figure.Figure object or str, the figure to be saved or path of the image to be loaded and saved 
           folder = str, folder where the image is saved
                    default: None, if not specified an Exception is raised
           output_name = str, name used for the image to be saved
                         default: None, if not set an Exception is raised
           output_path = str, general path where image is saved
                         default: r'images\report'
           dpi = int, dots per inch, i.e., image resolution
                 default: 600
           format = str, format for the image to be saved
                    default: 'png'
           cmap = str, key to set the visualization channels
                  default: 'gray', grayscale visualisation
           vmin = int, minimum value needed for visualisation
                  default: 0, can range from 0 to 255. 
           vmax = int, maximum value needed for visualisation
                  default: 1, can range from 0 to 255. 
    
    Output:
           None, it saves the image given the specified path, names and additional arguments
    '''
    if folder is None or output_name is None:
        raise Exception(f'You must specify `folder` and/or `output_name`.')
        
    save_path=os.path.join(output_path, folder, rf'{output_name}.{format}')
    if isinstance(fig, str): # load .tif image and update fig argument
       _, fig, _ = show_image_array(fig, scaled_classes=True, cmap=cmap, vmin=vmin, vmax=vmax, show=True, save_img=True)
           
    fig.savefig(save_path, format=format, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # close the figure to free up memory

    return None