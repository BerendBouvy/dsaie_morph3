import numpy as np
import pandas as pd
import argparse
from functions import *

def main(input_file, output_file, distance_limit, pixel_size):
    # Load the input images as 2D numpy arrays
    data = np.load(input_file)
