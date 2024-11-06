#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import glob
import csv
import SimpleITK as sitk
from scipy.spatial import distance
from scipy.ndimage import zoom
from numba import jit
import hashlib
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(22)
import torchmetrics
import io
from tqdm import tqdm
from functools import lru_cache

# Read GPU settings from environment
use_gpu = os.getenv('USE_GPU', 'true').lower() in ['true', '1', 't']
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

# Get anon_ids for which a segmentation file is present
#@jit(nopython=False, parallel=True, forceobj=True)
@lru_cache(maxsize=2**15)
def my_glob(search_path):
    return glob.glob(search_path)
def get_anon_ids_with_segmentation_files(anon_ids):

    dataset_dir = os.environ.get('DATASET_DIR', '/data/')

    anon_ids_filtered = []
    segmentation_files = []

    for anon_id in anon_ids:
        search_path = dataset_dir + anon_id + "-ENT.nii.gz"

        found_nifti_files = my_glob(search_path)

        if len(found_nifti_files) > 0:
            anon_ids_filtered.append(anon_id)
            segmentation_files.append(found_nifti_files[0])

    return anon_ids_filtered, segmentation_files


# Get anon_ids from dataset where a resection took place
#@jit(nopython=False, parallel=False, forceobj=True)
def get_anon_ids_and_filepaths():

    anon_ids = []

    df = pd.read_csv(os.environ.get(
        'DATASET_CSV', '/data/test_dataset.csv'), sep=",")

    df_resection = df[df['surgeryextend'] == 'resection']

    anon_id_column = 'AnonID'

    if 'anonID' in df.columns:
        anon_id_column = 'anonID'

    anon_ids = df_resection[anon_id_column].tolist()
    anon_ids, segmentation_files = get_anon_ids_with_segmentation_files(anon_ids)

    return anon_ids, segmentation_files


# Get distance matrix from disk
#@jit(nopython=False, parallel=False, forceobj=True)
def get_distance_matrix_from_file(input_file_path):

    distance_matrix = {}

    reader = csv.reader(open(input_file_path, 'r'))

    for row in reader:
       k, v = row
       distance_matrix[k] = v

    return distance_matrix


# Add distance if distance score is not present in file
#@jit(nopython=False, parallel=False, forceobj=True)
def compute_new_distances(input_array, input_file_path,torch_dict):
    anon_ids, segmentation_files = get_anon_ids_and_filepaths()

    distance_matrix = get_distance_matrix_from_file(input_file_path)

    distance_list = []

    for idx, anon_id in enumerate(anon_ids):
        if not anon_id in distance_matrix.keys():

            segmentation_file = segmentation_files[idx]
            if not( segmentation_file in torch_dict.keys()):
                # Read segmentation_array
                segmentation_image = sitk.ReadImage(segmentation_file)
                segmentation_array = sitk.GetArrayFromImage(segmentation_image)

                # Make sure segmentation_array is binary
                segmentation_array[segmentation_array == np.nan] = 0
                segmentation_array[segmentation_array > 0] = 1
                segmentation_array = torch.Tensor(segmentation_array).to(device).to(torch.bool)
                torch_dict[segmentation_file] = segmentation_array
            else:
                segmentation_array = torch_dict[segmentation_file]

            # Compute distance
            distance_score = torchmetrics.functional.dice(input_array.flatten(),segmentation_array.flatten(),ignore_index=0).cpu().item()

            distance_entry = str(anon_id) + ", " + str(distance_score)
            distance_list.append(distance_entry)

    return distance_list


# Write distances to distance file
def add_lines_to_file(distance_list, input_file_path):

    with open(input_file_path, 'a') as file:
        for line in distance_list:
            file.write(line + "\n")


# Return down sampled version of array
#@jit(nopython=False, parallel=False)
def down_sample_array(input_array):

    down_sampled_array = zoom(input_array, (0.25, 0.25, 0.25))

    return down_sampled_array


# Order distance_matrix by similarity
def order_by_similarity(distance_matrix):

    distance_matrix = sorted(
        distance_matrix.items(), key=lambda x: x[1])

    return distance_matrix


def get_hash(path, hash_type='sha256'):
    func = getattr(hashlib, hash_type)()
    f = os.open(path, os.O_RDWR )
    for block in iter(lambda: os.read(f, 2048*func.block_size), b''):
        func.update(block)
    os.close(f)
    return func.hexdigest()

# Get or create distance_matrix for input_array
#@jit(nopython=False, parallel=False, forceobj=True)
def get_or_create_distance_matrix(input_array,torch_dict,mask_path):

    # Make sure input is binary array
    input_array[input_array == np.nan] = 0
    input_array[input_array > 0] = 1

    # Downsample the input
    # down_sampled_input = down_sample_array(input_array)

    input_hash = get_hash(path=mask_path)

    if isinstance(input_array, np.ndarray):
        input_array = torch.Tensor(input_array).to(device).to(torch.bool)
    else:
        input_array = torch.Tensor(input_array.float()).to(device).to(torch.bool)

    distances_dir = os.environ.get('DATASET_DIR', '/data/') + "distances/"

    if not os.path.exists(distances_dir):
        os.makedirs(distances_dir)

    input_file_name = input_hash + ".csv"
    input_file_path = distances_dir + input_file_name

    array_list = glob.glob(input_file_path)
    distance_list = []

    if len(array_list) == 0:
        f= open(input_file_path,"w+")
        f.close()

    distance_list = compute_new_distances(input_array, input_file_path, torch_dict)

    if len(distance_list) > 0:
        add_lines_to_file(distance_list, input_file_path)

    distance_matrix = get_distance_matrix_from_file(input_file_path)
    distance_matrix = order_by_similarity(distance_matrix)

    return distance_matrix
