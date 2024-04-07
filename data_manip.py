import numpy as np
import pandas as pd
import matplotlib as mpl
import os 
from pyntcloud import PyntCloud
import time

# start_time = time.time()

class preprocessor():
    def __init__(self, input_path):
        self.coord_paths = []
        self.label_paths = []

        error = self.collect_paths(input_path)
        if error > 0:
            exit("***ERROR: NOT ENOUGH LABEL FILES FOUND FOR COORD FILES... TERMINATING")
        elif error < 0:
            exit("***ERROR: MORE LABEL FILES FOUND THAN COORD FILES... TERMINATING")

        print(f"\n{len(self.coord_paths)} TXT FILES: ")
        print("----------------")
        print("\n".join(self.coord_paths))
        print(f"\n{len(self.label_paths)} LABELS FILES: ")
        print("----------------")
        print("\n".join(self.label_paths))

        self.generate_matrices()


    def collect_paths(self, input_path):
        for root, dirs, files in os.walk(input_path):
            for name in files:
                if name.lower().endswith(".txt"):
                    self.coord_paths.append(os.path.join(root, name))
            for sub_dir in dirs:
                for label_root, _, label_files in os.walk(os.path.join(root,sub_dir)):
                    for labels in label_files:
                        if labels.lower().endswith(".labels"):
                            self.label_paths.append(os.path.join(label_root, labels))
        return len(self.coord_paths) - len(self.label_paths)


    def generate_matrices(self):
        folder_path = "preprocessed_data\\"
        os.makedirs(folder_path, exist_ok=True)
        processed_list = [np.empty((0,0,0))]*9
        for scene in range(len(self.coord_paths)): #len(self.coord_paths)
            coords = np.genfromtxt(self.coord_paths[scene], dtype=float,usecols=(0,1,2))
            labels = np.genfromtxt(self.label_paths[scene], dtype=int)
            for i in range(9):
                # Extracting points associated with current feature
                mask = (labels == i)
                pts = coords[mask]
                # Sampling 2048 points
                if len(pts) > 2048:
                    pts = pts[np.random.choice(pts.shape[0], size = 2048)]
                elif len(pts) < 2048:
                    pad_width = ((0,1),(0,0))
                    while len(pts) < 2048:
                        pts = np.pad(pts,pad_width,mode='constant',constant_values=0)
                # Normalizing points to [-1,1]
                pts = 2.*(pts-np.min(pts,axis=0))/np.ptp(pts,axis=0)-1 #pts = pts / np.max(np.abs(pts),axis=0) * 2

                if processed_list[i].size == 0:
                    processed_list[i] = pts[np.newaxis,:]
                    print(np.shape(processed_list[i]))
                else:
                    pts = pts[np.newaxis,:]
                    processed_list[i] = np.concatenate((processed_list[i],pts), axis=0)

        for matrix in range(len(processed_list)):
            file_path = os.path.join(folder_path, f"{str(matrix)}.npy")
            np.save(file_path,processed_list[matrix])
                




mypreprocessor = preprocessor("input\\")
# data = np.load("C:\\Users\\ofwol\\Documents\\PythonProjects\\Terrain3DPointCloudGeneration\\data\\chair_test.npy")
# print(np.shape(data))


# data_path = "C:\\Users\\ofwol\\Documents\\PythonProjects\\765\\input\\bildstein_station1_xyz_intensity_rgb\\bildstein_station1_xyz_intensity_rgb.txt"
# label_path = "C:\\Users\\ofwol\\Documents\\PythonProjects\\765\\input\\sem8_labels_training\\bildstein_station1_xyz_intensity_rgb.labels"
# class PointCloud():
#     def __init__(self, data_path, label_path):
#         unlabeled = []
#         man_made = []
#         natural = []
#         high_veg =[]
#         low_veg =[]
#         buildings =[]
#         hard_scape =[]
#         artefacts =[]
#         cars =[]

#         self.dataframes = []

#         with open(label_path) as label_file, open(data_path) as coord_file:
#             for label_line, coord_line in zip(label_file, coord_file):
#                 label = int(label_line.strip())  # Convert label to an integer
#                 x, y, z, i, r, g, b = map(float, coord_line.strip().split())  # Extract coordinates
#                 values = [x, y, z]
#                 match label:
#                     case 0:
#                         unlabeled.append(values)
#                     case 1:
#                         man_made.append(values)
#                     case 2:
#                         natural.append(values)
#                     case 3:
#                         high_veg.append(values)
#                     case 4:
#                         low_veg.append(values)
#                     case 5:
#                         buildings.append(values)
#                     case 6:
#                         hard_scape.append(values)
#                     case 7:
#                         artefacts.append(values)
#                     case 8:
#                         cars.append(values)

#         self.unlabeled_df = pd.DataFrame(unlabeled, columns=["x","y","z"])
#         #self.unlabeled_df.style.set_caption("unlabeled")
#         self.man_made_df = pd.DataFrame(man_made, columns=["x","y","z"])
#         #self.man_made_df.style.set_caption("man_made")
#         self.natural_df = pd.DataFrame(natural, columns=["x","y","z"])
#         #self.natural_df.style.set_caption("natural")
#         self.high_veg_df = pd.DataFrame(high_veg, columns=["x","y","z"])
#         #self.high_veg_df.style.set_caption("high_veg")
#         self.low_veg_df = pd.DataFrame(low_veg, columns=["x","y","z"])
#         #self.low_veg_df.style.set_caption("low_veg")
#         self.buildings_df = pd.DataFrame(buildings, columns=["x","y","z"])
#         #self.buildings_df.style.set_caption("buildings")
#         self.hard_scape_df = pd.DataFrame(hard_scape, columns=["x","y","z"])
#         #self.hard_scape_df.style.set_caption("hard_scape")
#         self.artefacts_df = pd.DataFrame(artefacts, columns=["x","y","z"])
#         #self.artefacts_df.style.set_caption("artefacts")
#         self.cars_df = pd.DataFrame(cars, columns=["x","y","z"])
#         #self.cars_df.style.set_caption("cars")

#         self.dataframes.append(self.unlabeled_df)
#         self.dataframes.append(self.man_made_df)
#         self.dataframes.append(self.natural_df)
#         self.dataframes.append(self.high_veg_df)
#         self.dataframes.append(self.low_veg_df)
#         self.dataframes.append(self.buildings_df)
#         self.dataframes.append(self.hard_scape_df)
#         self.dataframes.append(self.artefacts_df)
#         self.dataframes.append(self.cars_df)

        
#         #NORMALIZING DATA TO -1 to 1
#         for df in self.dataframes:
#             for col in df.columns:
#                 df[col] = self.abs_max_scale(df[col])

#         self.buildings_df.to_pickle("pickles\\buildings.pkl")

#     def plot_segments(self, id, s):
#         match id:
#             case 0:
#                 cloud = PyntCloud(self.unlabeled_df.sample(s))
#             case 1:
#                 cloud = PyntCloud(self.man_made_df.sample(s))
#             case 2:
#                 cloud = PyntCloud(self.natural_df.sample(s))
#             case 3:
#                 cloud = PyntCloud(self.high_veg_df.sample(s))
#             case 4:
#                 cloud = PyntCloud(self.low_veg_df.sample(s))
#             case 5:
#                 cloud = PyntCloud(self.buildings_df.sample(s))
#             case 6:
#                 cloud = PyntCloud(self.hard_scape_df.sample(s))
#             case 7:
#                 cloud = PyntCloud(self.artefacts_df.sample(s))
#             case 8:
#                 cloud = PyntCloud(self.cars_df.sample(s))
#         cloud.plot(width=1000, height=1000, backend="matplotlib")

#     def abs_max_scale(ind,col):
#         return col / col.abs().max()

# myCloud = PointCloud(data_path, label_path)
# end_time = time.time()

# # Calculate the runtime
# runtime_seconds = end_time - start_time
# runtime_minutes = runtime_seconds / 60

# print(f"Total Parse Time: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")

# while True:
#     id = input("Enter ID of points to display, 'q' to quit: ")
#     if id == 'q':
#         break
#     sample = input("number of points to sample, 'q' to quit: ")
#     if sample == 'q':
#         break
#     else:
#         myCloud.plot_segments(int(id), int(sample))