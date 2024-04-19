import numpy as np
import pandas as pd
import matplotlib as mpl
import os 
import time

# start_time = time.time()

class preprocessor():
    def __init__(self, input_path):
        self.input_path = input_path
        self.coord_paths = self.collect_paths(input_path)

    def collect_paths(self, input_path):
        coord_paths = []
        label_paths = []
        for root, dirs, files in sorted(os.walk(input_path)):
            for name in files:
                if name.lower().endswith(".xyz"):
                    coord_paths.append(os.path.join(root, name))
                if name.lower().endswith(".labels"):
                    label_paths.append(os.path.join(root, name))
        
        """  error = len(coord_paths) - len(label_paths)
        if error > 0:
            exit("***ERROR: NOT ENOUGH LABEL FILES FOUND FOR COORD FILES... TERMINATING")
        elif error < 0:
            exit("***ERROR: MORE LABEL FILES FOUND THAN COORD FILES... TERMINATING") """

        print(f"\n{len(coord_paths)} .XYZ FILES: ")
        print("----------------")
        print("\n".join(coord_paths))
        print(f"\n{len(label_paths)} LABELS FILES: ")
        print("----------------")
        print("\n".join(label_paths))
        return coord_paths
    
    def normalize(self,data):
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

    def generate_matrices(self, num_coordinates = 2048, num_samples = 32, set_name = 'train',feature = 5):
        matrix_output_folder = self.input_path + "/matrices"
        os.makedirs(matrix_output_folder, exist_ok=True)
        point_cloud_objects = []
        for scene in range(len(self.coord_paths)): #len(self.coord_paths)
            #name = self.coord_paths[scene].split("/")[-1].split(".")[0]
            #raw_point_cloud = np.genfromtxt(self.coord_paths[scene], dtype=float,usecols=(0,1,2))
            #labels = np.genfromtxt(self.label_paths[scene], dtype=int)
            raw_point_cloud = pd.read_csv(self.coord_paths[scene], usecols=[0,1,2], delimiter= " ", header=None).values
            """label_file_name = self.coord_paths[scene].split(".")[0] + ".labels"
            if not os.path.exists(label_file_name):
                print(f"Labels file {label_file_name} not found")
                continue
            labels = pd.read_csv(label_file_name, header = None).values"""
            print(np.shape(raw_point_cloud))
            #print(np.shape(labels)) 

            ### Extracting points associated with feature
            
            """label_mask_1d = (labels == feature)
            label_mask_2d = np.repeat(label_mask_1d, raw_point_cloud.shape[1]).reshape(-1,raw_point_cloud.shape[1])
            label_point_cloud = raw_point_cloud[label_mask_2d].reshape(-1,raw_point_cloud.shape[1])"""
            
            ### Sampling {sample_size} points
            
            if len(raw_point_cloud) > num_coordinates:
                for sample_number in range(num_samples):
                    random_indices = np.random.choice(raw_point_cloud.shape[0], num_coordinates, replace=False)
                    point_cloud_object = raw_point_cloud[random_indices]
                    #print(point_cloud_object.shape)
                    point_cloud_objects.append(point_cloud_object)
                    #normalized_point_cloud_object = 2.*(point_cloud_object-np.min(point_cloud_object,axis=0))/np.ptp(point_cloud_object,axis=0)-1
                    #point_cloud_objects[sample_number] = normalized_point_cloud_object
                                
            """elif len(raw_point_cloud) < num_coordinates:
                print(f'{self.coord_paths[scene]} has less than {num_coordinates} points for label {feature}... padding')
                for sample_number in range(num_samples):
                    point_cloud_object = np.pad(raw_point_cloud, ((0,num_coordinates-len(raw_point_cloud)),(0,0)), 'wrap')
                    point_cloud_objects.append(point_cloud_object)
                    #normalized_point_cloud_object = 2.*(point_cloud_object-np.min(point_cloud_object,axis=0))/np.ptp(point_cloud_object,axis=0)-1"""
            
        ### stack the 2d arrays into one giant 3d array
        stacked_point_cloud_objects = np.stack(point_cloud_objects)
        print(stacked_point_cloud_objects.shape)
        normalized_stacked_point_cloud_objects = stacked_point_cloud_objects
        for j in range(normalized_stacked_point_cloud_objects.shape[2]):
            normalized_stacked_point_cloud_objects[:,:,j] = self.normalize(normalized_stacked_point_cloud_objects[:,:,j])
        print(normalized_stacked_point_cloud_objects.shape)
        name_table = {0:"unlabeled", 1:"man_made", 2:"natural", 3:"high_veg",4:"low_veg",5:"buildings",6:"hard_scape",7:"artifacts",8:"cars"}
        np.save(f"building3d_{num_coordinates}_{num_samples}_{set_name}.npy", normalized_stacked_point_cloud_objects, allow_pickle=False)
                
            


mypreprocessor = preprocessor("/share/lsmsmart/regarry/Terrain3DPointCloudGeneration/data/building3d/pointcloud")
mypreprocessor.generate_matrices(2048, 1, 'train')
#mypreprocessor.generate_matrices(2048, 100, 'test', feature = 5)
