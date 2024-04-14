import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description='Process some NPY.')
parser.add_argument('-d', dest='datafile', type=str, help='The data file path')
parser.add_argument('-e', dest='elevation', type=int, default=70, help='Elevation angle')
parser.add_argument('-a', dest='azimuth', type=int, default=30, help='Azimuth angle')

args = parser.parse_args()
data = np.load(args.datafile)
print(data.shape)
if data.shape[2] == 3:
    cloud = data[0,:,:]
else:
    cloud = data[0,:,:].T
print(cloud.shape)

# Create a new matplotlib figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using the x, y, z attributes of the point cloud
ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2])

# Set the viewing angle
ax.view_init(elev=args.elevation, azim=args.azimuth)


# Save the plot as a PNG image
folder = "visuals"
filename = os.path.basename(args.datafile)
if not os.path.exists(folder):
    os.makedirs(folder)
plt.savefig(f"{folder}/{filename}_e_{args.elevation}_a_{args.azimuth}.png")