import numpy as np
import pandas as pd
import matplotlib as mpl
# import open3d as open
import os 
from pyntcloud import PyntCloud
import time

start_time = time.time()

data_path = "C:\\Users\\ofwol\\Documents\\PythonProjects\\765\\input\\bildstein_station1_xyz_intensity_rgb\\bildstein_station1_xyz_intensity_rgb.txt"
label_path = "C:\\Users\\ofwol\\Documents\\PythonProjects\\765\\input\\sem8_labels_training\\bildstein_station1_xyz_intensity_rgb.labels"

# building_pts = []

# with open(label_path) as label_file, open(data_path) as coord_file:
#     for label_line, coord_line in zip(label_file, coord_file):
#         label = int(label_line.strip())  # Convert label to an integer
#         x, y, z, i, r, g, b = map(float, coord_line.strip().split())  # Extract coordinates

#         if label == 5:
#             values = [x, y, z]
#             building_pts.append(values)

# buildings_df = pd.DataFrame(building_pts, columns=["x","y","z"])
# end_time = time.time()

# # Calculate the runtime
# runtime_seconds = end_time - start_time
# runtime_minutes = runtime_seconds / 60

# print(f"Total runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")


# df_data = pd.read_csv(data_path, sep=" ", header=None, names=["x", "y", "z", "intensity", "r", "g", "b"])
# df_labels = pd.read_csv(label_path, sep=" ",header=None, names=["label"])
# cloud = PyntCloud(buildings_df.sample(2048))
# cloud.plot()

class PointCloud():
    def __init__(self, data_path, label_path):
        unlabeled = []
        man_made = []
        natural = []
        high_veg =[]
        low_veg =[]
        buildings =[]
        hard_scape =[]
        artefacts =[]
        cars =[]

        self.dataframes = []

        with open(label_path) as label_file, open(data_path) as coord_file:
            for label_line, coord_line in zip(label_file, coord_file):
                label = int(label_line.strip())  # Convert label to an integer
                x, y, z, i, r, g, b = map(float, coord_line.strip().split())  # Extract coordinates
                values = [x, y, z]
                match label:
                    case 0:
                        unlabeled.append(values)
                    case 1:
                        man_made.append(values)
                    case 2:
                        natural.append(values)
                    case 3:
                        high_veg.append(values)
                    case 4:
                        low_veg.append(values)
                    case 5:
                        buildings.append(values)
                    case 6:
                        hard_scape.append(values)
                    case 7:
                        artefacts.append(values)
                    case 8:
                        cars.append(values)

        self.unlabeled_df = pd.DataFrame(unlabeled, columns=["x","y","z"])
        #self.unlabeled_df.style.set_caption("unlabeled")
        self.man_made_df = pd.DataFrame(man_made, columns=["x","y","z"])
        #self.man_made_df.style.set_caption("man_made")
        self.natural_df = pd.DataFrame(natural, columns=["x","y","z"])
        #self.natural_df.style.set_caption("natural")
        self.high_veg_df = pd.DataFrame(high_veg, columns=["x","y","z"])
        #self.high_veg_df.style.set_caption("high_veg")
        self.low_veg_df = pd.DataFrame(low_veg, columns=["x","y","z"])
        #self.low_veg_df.style.set_caption("low_veg")
        self.buildings_df = pd.DataFrame(buildings, columns=["x","y","z"])
        #self.buildings_df.style.set_caption("buildings")
        self.hard_scape_df = pd.DataFrame(hard_scape, columns=["x","y","z"])
        #self.hard_scape_df.style.set_caption("hard_scape")
        self.artefacts_df = pd.DataFrame(artefacts, columns=["x","y","z"])
        #self.artefacts_df.style.set_caption("artefacts")
        self.cars_df = pd.DataFrame(cars, columns=["x","y","z"])
        #self.cars_df.style.set_caption("cars")

        self.dataframes.append(self.unlabeled_df)
        self.dataframes.append(self.man_made_df)
        self.dataframes.append(self.natural_df)
        self.dataframes.append(self.high_veg_df)
        self.dataframes.append(self.low_veg_df)
        self.dataframes.append(self.buildings_df)
        self.dataframes.append(self.hard_scape_df)
        self.dataframes.append(self.artefacts_df)
        self.dataframes.append(self.cars_df)

        
        #NORMALIZING DATA TO -1 to 1
        for df in self.dataframes:
            for col in df.columns:
                df[col] = self.abs_max_scale(df[col])

        self.buildings_df.to_pickle("pickles\\buildings.pkl")

    def plot_segments(self, id, s):
        match id:
            case 0:
                cloud = PyntCloud(self.unlabeled_df.sample(s))
            case 1:
                cloud = PyntCloud(self.man_made_df.sample(s))
            case 2:
                cloud = PyntCloud(self.natural_df.sample(s))
            case 3:
                cloud = PyntCloud(self.high_veg_df.sample(s))
            case 4:
                cloud = PyntCloud(self.low_veg_df.sample(s))
            case 5:
                cloud = PyntCloud(self.buildings_df.sample(s))
            case 6:
                cloud = PyntCloud(self.hard_scape_df.sample(s))
            case 7:
                cloud = PyntCloud(self.artefacts_df.sample(s))
            case 8:
                cloud = PyntCloud(self.cars_df.sample(s))
        cloud.plot(width=1000, height=1000, backend="matplotlib")

    def abs_max_scale(ind,col):
        return col / col.abs().max()

myCloud = PointCloud(data_path, label_path)
end_time = time.time()

# Calculate the runtime
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60

print(f"Total Parse Time: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

while True:
    id = input("Enter ID of points to display, 'q' to quit: ")
    if id == 'q':
        break
    sample = input("number of points to sample, 'q' to quit: ")
    if sample == 'q':
        break
    else:
        myCloud.plot_segments(int(id), int(sample))