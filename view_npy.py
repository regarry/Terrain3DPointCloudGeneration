import numpy as np
import argparse


if __name__ == '__main__':
    ### parse arguments for data file path
    parser = argparse.ArgumentParser(description='Process some NPY files.')
    parser.add_argument('-d', dest='datafile', type=str, help='The data file path')

    # Parse the arguments
    args = parser.parse_args()

    # Load the data
    data = np.load(args.datafile)

    # Print the type, shape, and first few elements of the data
    print("Type: ", type(data))
    print("Shape: ", data.shape)
    print("First few elements: ", data[:10])