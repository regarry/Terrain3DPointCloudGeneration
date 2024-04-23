import numpy as np
import pandas as pd
import matplotlib as mpl
import os 
import time
import random

# start_time = time.time()

class preprocessor():
    def __init__(self, input_path):
        self.input_path = input_path
        self.coord_paths = self.collect_paths(input_path)
        self.split_paths()
    
    def collect_paths(self, input_path):
        coord_paths = []
        for root, dirs, files in sorted(os.walk(input_path)):
            for name in files:
                if name.lower().endswith(".xyz"):
                    coord_paths.append(os.path.join(root, name))
        
        print(f"\n{len(coord_paths)} .XYZ FILES: ")
        print("----------------")
        print("\n".join(coord_paths))
        return coord_paths
    
    def normalize(self,data):
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
    
    def split_paths(self, test_ratio=0.1, shuffle_paths=False):
            num_paths = len(self.coord_paths)
            num_test = int(test_ratio * num_paths)
            num_train = num_paths - num_test
            
            if shuffle_paths:
                random.shuffle(self.coord_paths)
            
            self.train_paths = self.coord_paths[:num_train]
            self.test_paths = self.coord_paths[num_train:]
            return None
    
    def generate_matrices(self, num_coordinates = 2048, num_samples = 32, set_name = 'train',feature = 5):
        skip_count = 0
        matrix_output_folder = self.input_path + 'output'
        os.makedirs(matrix_output_folder, exist_ok=True)
        point_cloud_objects = []
        file_paths = self.train_paths if set_name == 'train' else self.test_paths
        for file in file_paths: 
            raw_point_cloud = pd.read_csv(file, usecols=[0,1,2], delimiter= " ", header=None).values
            #print(np.shape(raw_point_cloud))


            ### Sampling {sample_size} points
            
            if len(raw_point_cloud) > num_coordinates:
                for sample_number in range(num_samples):
                    random_indices = np.random.choice(raw_point_cloud.shape[0], num_coordinates, replace=False)
                    point_cloud_object = raw_point_cloud[random_indices]
                    #print(point_cloud_object.shape)
                    point_cloud_objects.append(point_cloud_object)
                    #normalized_point_cloud_object = 2.*(point_cloud_object-np.min(point_cloud_object,axis=0))/np.ptp(point_cloud_object,axis=0)-1
                    #point_cloud_objects[sample_number] = normalized_point_cloud_object
                                
            elif len(raw_point_cloud) < num_coordinates:
                skip_count += 1
                """print(f'{file} has less than {num_coordinates} points for label {feature}... padding')
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
        print(f'{skip_count} files had less than {num_coordinates} points and were skipped')
        #name_table = {0:"unlabeled", 1:"man_made", 2:"natural", 3:"high_veg",4:"low_veg",5:"buildings",6:"hard_scape",7:"artifacts",8:"cars"}
        np.save(f"building3d_{num_coordinates}_{num_samples}_{set_name}.npy", normalized_stacked_point_cloud_objects, allow_pickle=False)
                
            
mypreprocessor = preprocessor("/share/lsmsmart/regarry/Terrain3DPointCloudGeneration/data/building3d/pointclouds/")
mypreprocessor.generate_matrices(2048, 1, 'train')
mypreprocessor.generate_matrices(2048, 1, 'test')
