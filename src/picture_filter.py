#!/usr/bin/env python3
import logging
import os
import numpy as np
import pandas as pd
import time
import random
import glob
import SimpleITK as sitk
import datatype_helper
import custom_mappings
import base64
import uuid
from compute_distances import get_or_create_distance_matrix
from numba import jit
from functools import lru_cache
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(22)
from tqdm import tqdm


torch_dict = {}

# Function to detect the available device (GPU or CPU)
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()
torch.set_grad_enabled(False)

# Function to load a NIFTI image as a numpy array and cache it
# This function also reshapes the image if necessary
logging.info(f"Running on device: {device}")

def get_image_array(image_path, reshape=False):
    global torch_dict
    if not(image_path in torch_dict.keys()):
        image = sitk.ReadImage(image_path)

        if reshape != False:
            image.SetOrigin(reshape.GetOrigin())
            image.SetSpacing(reshape.GetSpacing())
            image.SetDirection(reshape.GetDirection())

        image_array = sitk.GetArrayFromImage(image)

        # Convert the array to a PyTorch tensor and move it to the detected device
        image_array = torch.Tensor(image_array).to(device).to(torch.bool)
        torch_dict[image_path] = image_array
    else:
        image_array = torch_dict[image_path]

    return image_array

# Function to load a numpy array from a binary file
def load_numpy_array(image_path):
    image_array = np.load(image_path)
    return image_array

# Get sub-image from a file path based on mask image and mask indices
def get_sub_image_from_file_path(file_path, mask_image, mask_indices):
    image_array = get_image_array(file_path, mask_image)
    #sub_image_array = get_sub_image(image_array, mask_indices)
    return image_array

# Check if scans are present for a given anon_id
@lru_cache(maxsize=4096)
def check_data_entry(anon_id):
    dataset_dir = os.environ.get('DATASET_DIR', '/data/')

    scan_type_mapping = {
        'segmentation': 'ENT',
        'residue': 'ENR'
    }

    results = []

    for scan_type, label in scan_type_mapping.items():
        search_path = dataset_dir + anon_id + "-*[0-9]*-" +  label + ".nii.gz"
        found_nifti_files = glob.glob(search_path)

        if(len(found_nifti_files) > 0):
            results.append(found_nifti_files[0])

    if (len(results) == 2):
        return results
    else:
        return [None, None]

# Function to retrieve supported filter criteria from dataset CSV
def get_supported_filter_criteria():
    filter_criteria = []

    df = pd.read_csv(os.environ.get(
        'DATASET_CSV', '/data/test_dataset.csv'), sep=",")

    # Add birthyears to the dataset
    df = custom_mappings.add_birthyear(df)
    filter_criteria = df.columns.tolist()[1:]

    supported_filter_criteria = []
    filter_criteria_mapping = custom_mappings.get_filter_criteria_mapping(
        inverted=True)

    for i in filter_criteria:
        if i in filter_criteria_mapping.keys():
            supported_filter_criteria.append(filter_criteria_mapping[i])

    return supported_filter_criteria

# Function to get filterable columns based on supported filter criteria
def get_filterable_columns():
    supported_filter_criteria = get_supported_filter_criteria()
    filter_criteria_mapping = custom_mappings.get_filter_criteria_mapping()

    filterable_columns = []

    for i in supported_filter_criteria:
        if i in filter_criteria_mapping.keys():
            filterable_columns.append(filter_criteria_mapping[i])

    return filterable_columns

# Function to serialize a dataset for client representation
def get_serialized_data_set(data_set, filterable_output):
    filter_criteria_mapping = custom_mappings.get_filter_criteria_mapping(
        inverted=True)
    data_type_mapping = datatype_helper.get_data_type_mapping()

    for k in list(filterable_output.keys()):
        np_data_type = data_set[k].dtype

        for i, j in data_type_mapping.items():
            if np_data_type == i:
                data_type = j
                break

        options = {}
        values = data_set[k]

        options, values, data_type, upper_boundary = custom_mappings.get_column_options_and_values(
            data_type, values, filter_criteria_mapping, k)

        result = {
            'values': values,
            'data_type': data_type,
            'options': options
        }

        if data_type == 'float' or data_type == 'integer':
            if upper_boundary:
                result['upper_boundary'] = upper_boundary

        filterable_output.pop(k)
        filterable_output[filter_criteria_mapping[k]] = result

    return filterable_output

# Function to get filterable dataframe and selection based on criteria
def get_clinical_variables(
    data_frame, data_frame_filtered, supported_filter_criteria, filter_criteria={}):

    # Get filterable output
    filterable_columns = get_filterable_columns()
    filterable_output = dict.fromkeys(filterable_columns, None)
    filterable_output = get_serialized_data_set(data_frame, filterable_output)

    # Get selection_output
    selection_output = dict.fromkeys(filterable_columns, None)
    selection_output = get_serialized_data_set(data_frame_filtered, selection_output)

    result_data = {
        'filterable_output': filterable_output,
        'selection': selection_output
    }

    return result_data

# Function to apply filters to a dataframe
def apply_filters(df, filter_criteria):
    valid_filter_criteria = df.columns.tolist()[1:]
    filter_criteria_mapping = custom_mappings.get_filter_criteria_mapping()
    binned_columns = custom_mappings.get_binned_columns()

    for k, v in filter_criteria.items():
        k = filter_criteria_mapping[k]

        if not k in valid_filter_criteria:
            continue

        if not isinstance(v, list):
            v = [v]

        filter_values = []

        if len(v) == 1:
            filter_values = v
        elif len(v) > 1:
            filter_min = v[0]
            filter_max = v[1]

            if len(v) == 3:
                include_unknown = v[2]
                if include_unknown.lower() == "true":
                    include_unknown = True
                else:
                    include_unknown = False
            else:
                include_unknown = False

            apply_min_max = False

            if df.dtypes[k] == np.int64:
                if datatype_helper.is_float(filter_min):
                    filter_min = datatype_helper.cast_to_digit(filter_min, True)
                else:
                    continue

                if datatype_helper.is_float(filter_max):
                    filter_max = datatype_helper.cast_to_digit(filter_max, True)
                else:
                    continue
                apply_min_max = True
            elif df.dtypes[k] == np.float64:
                if datatype_helper.is_float(filter_min):
                    filter_min = datatype_helper.cast_to_digit(filter_min)
                else:
                    continue

                if datatype_helper.is_float(filter_max):
                    filter_max = datatype_helper.cast_to_digit(filter_max)
                else:
                    continue
                apply_min_max = True

            if apply_min_max:
                if not k in binned_columns:
                    df = df[(df[k] >= filter_min) & (df[k] <= filter_max)]
                else:
                    values_in_bin = custom_mappings.get_values_in_bin_range(
                        k, df[k], filter_min, filter_max, include_unknown)

                    df = df[df[k].isin(values_in_bin)]
            else:
                filter_values = v

        casted_filter_values = []

        for filter_val in filter_values:
            if df.dtypes[k] == np.int64:
                if datatype_helper.is_float(filter_val):
                    filter_val = datatype_helper.cast_to_digit(filter_val, True)
                else:
                    continue

            elif df.dtypes[k] == np.float64:
                if datatype_helper.is_float(filter_val):
                    filter_val = datatype_helper.cast_to_digit(filter_val)
                else:
                    continue

            elif df.dtypes[k] == np.bool:
                filter_val = datatype_helper.cast_to_boolean(filter_val)

            elif df.dtypes[k] == np.object:
                clinical_variable_values = df[k].dropna().tolist()
                validation_values = [True, False]

                if all(elem in validation_values for elem in clinical_variable_values):
                    filter_val = datatype_helper.cast_to_boolean(filter_val)

            casted_filter_values.append(filter_val)

        if casted_filter_values:
            if not k in binned_columns:
                values_in_column = df[k].tolist()
                filterable_values = []

                for filter_val in casted_filter_values:
                    if filter_val in values_in_column:
                        filterable_values.append(filter_val)

                if filterable_values:
                    df = df[df[k].isin(filterable_values)]
            else:
                values_in_bin = []

                for filter_val in casted_filter_values:
                    values_in_bin.extend(custom_mappings.get_values_in_bin_range(
                        k, df[k], filter_val, include_unknown))

                df = df[df[k].isin(values_in_bin)]

    return df

# Function to get a filtered dataset of NIFTI files from a CSV mapping
def get_filtered_dataset(distance_matrix, filter_criteria={}):
    df = pd.read_csv(os.environ.get(
        'DATASET_CSV', '/data/test_dataset.csv'), sep=",")

    # Initialize new columns in the DataFrame for the file paths and similarity
    df['segmentation'] = None
    df['residue'] = None
    df['similarity'] = None

    anon_id_column = 'AnonID'
    if 'anonID' in df.columns:
        anon_id_column = 'anonID'

    # Filter dataset for resection entries
    df_resection = df[df['surgeryextend'] == 'resection']
    df_resection_filtered = apply_filters(df_resection, filter_criteria)
    anon_ids = df_resection_filtered[anon_id_column].tolist()

    for anon_id in anon_ids:
        segmentation, residue = check_data_entry(anon_id)

        if segmentation and residue:
            df.loc[df[anon_id_column] == anon_id, 'segmentation'] = segmentation
            df.loc[df[anon_id_column] == anon_id, 'residue'] = residue

            if distance_matrix:
                if anon_id in distance_matrix:
                    similarity = 1 - distance_matrix[anon_id]
                    df.loc[df[anon_id_column] == anon_id, 'similarity'] = similarity

    data_set = {}
    data_set['segmentation'] = []
    data_set['residue'] = []

    for i, j in data_set.items():
        data_set[i] = datatype_helper.convert_to_numba(df[i].dropna().tolist())

    return (data_set, df_resection, df_resection_filtered)

# Apply binary mask to the dataset results
def get_masked_results(input_array, binary_mask):
    input_array = input_array * binary_mask
    return input_array

# Calculate probability map for dataset based on binary mask
def get_probability_map(data_set, binary_mask):
    logging.info("Starting get_probability_map function")

    # Check data types
    logging.info(f"Data types - sum_residue: {data_set['sum_residue'].dtype}, sum_tumors: {data_set['sum_tumors'].dtype}")

    # Ensure sum_residue and sum_tumors are tensors
    assert isinstance(data_set['sum_residue'], torch.Tensor), "sum_residue is not a tensor"
    assert isinstance(data_set['sum_tumors'], torch.Tensor), "sum_tumors is not a tensor"

    # Ensure the data types are correct
    assert data_set['sum_residue'].dtype == torch.float32, f"Expected sum_residue dtype to be torch.float32 but got {data_set['sum_residue'].dtype}"
    assert data_set['sum_tumors'].dtype == torch.float32, f"Expected sum_tumors dtype to be torch.float32 but got {data_set['sum_tumors'].dtype}"

    logging.info("sum_residue and sum_tumors tensors have the correct types.")

    # Log tensor shapes
    logging.info(f"Shapes - sum_residue: {data_set['sum_residue'].shape}, sum_tumors: {data_set['sum_tumors'].shape}")

    # Apply the mask
    sum_residue_masked = torch.where(binary_mask, data_set['sum_residue'], torch.tensor(0.0, device=device))
    sum_tumors_masked = torch.where(binary_mask, data_set['sum_tumors'], torch.tensor(1.0, device=device))

    logging.info(f"sum_residue_masked and sum_tumors_masked created with shapes: {sum_residue_masked.shape}, {sum_tumors_masked.shape}")

    # Avoid division by zero
    with torch.no_grad():
        safe_sum_tumors_masked = torch.where(sum_tumors_masked == 0, torch.tensor(1.0, device=device), sum_tumors_masked)

    # Calculate the probability map
    probability_map = 1.0 - (sum_residue_masked / safe_sum_tumors_masked)

    # Set voxels where both sum_tumors_masked and sum_residue_masked are zero and within the masked area to 1.001
    # This ensures we only change values where both conditions are met and inside the mask
    condition = (sum_tumors_masked == 0) & (sum_residue_masked == 0) & binary_mask
    probability_map = torch.where(condition, torch.tensor(1.001, device=device), probability_map)

    logging.info(f"Probability map calculated with shape: {probability_map.shape}")

    return probability_map

# Convert dataset dictionary of lists to aggregated dataset
def aggregate_dataset(data_set, mask, mask_image, mask_indices):
    """
    Aggregates the dataset by applying the binary mask to the sum_tumors and sum_residue arrays.

    Args:
        data_set (dict): Dictionary containing 'segmentation' and 'residue' keys with lists of NIFTI file paths.
        mask (numpy.ndarray): Binary mask array.
        mask_image (SimpleITK.Image): Mask image for setting origin, spacing, and direction.
        mask_indices (numpy.ndarray): Indices for sub-image extraction.

    Returns:
        dict: Aggregated dataset with 'sum_tumors', 'sum_residue', and 'probability_map'.
    """
    logging.info("Starting aggregate_dataset function")

    # Define mapping for aggregation
    aggregate_mapping = {
        'segmentation': 'sum_tumors',
        'residue': 'sum_residue'
    }

    # Ensure the binary mask is boolean
    binary_mask = mask
    binary_mask[binary_mask > 0] = 1
    binary_mask = torch.tensor(binary_mask).to(device).to(torch.bool)  # Ensure it's boolean

    # Loop through the dataset items and perform aggregation
    for i, j in list(data_set.items()):
        logging.info(f"Aggregating data for {i}")

        union = False

        if aggregate_mapping[i] == 'sum_tumors':
            union = True

        # Sum array and apply tumor mask
        data_set[aggregate_mapping[i]] = get_masked_results(
            get_summed_array(j, mask_image, mask_indices, union), binary_mask)

        # Convert sum_residue to float32
        if aggregate_mapping[i] == 'sum_residue':
            data_set[aggregate_mapping[i]] = data_set[aggregate_mapping[i]].to(torch.float32)

        # Drop the individual data because of the memory footprint
        data_set.pop(i)

    # Add probability map
    data_set['probability_map'] = get_probability_map(data_set, binary_mask)

    return data_set

# Optimized function to get union array of two binary numpy arrays
def get_array_union(array_1, array_2):
    union_array = (array_1 + array_2) > 0
    return union_array

# Optimized function to get ENR file path from ENT file path
@lru_cache(maxsize=4096)
def get_enr_from_ent(ent_file_path):
    dataset_dir = os.environ.get('DATASET_DIR', '/data/')
    ent_filename = os.path.basename(ent_file_path)
    anon_id = ent_filename.split('-')[0]

    search_path = dataset_dir + anon_id + "-*[0-9]*-ENR.nii.gz"
    found_nifti_files = glob.glob(search_path)

    if(len(found_nifti_files) > 0):
        return found_nifti_files[0]
    else:
        return ""

# Optimized function to get summed array of list of nifti files
def get_summed_array(nifti_files, mask_image, mask_indices, union=False):
    summed_array = None

    for idx, nifti_file in enumerate(nifti_files):
        if idx < (len(nifti_files) - 1):
            input_1 = get_sub_image_from_file_path(
                nifti_file, mask_image, mask_indices) if idx == 0 else summed_array

            input_2 = get_sub_image_from_file_path(
                nifti_files[idx + 1], mask_image, mask_indices)

            # If union is true, get the ENR image as well
            if union == True:
                if idx == 0:
                    enr_nifti_file = get_enr_from_ent(nifti_file)
                    input_1_enr = get_sub_image_from_file_path(
                        enr_nifti_file, mask_image, mask_indices)
                    input_1 = get_array_union(input_1, input_1_enr).to(torch.float32)

                enr_2_nifti_file = get_enr_from_ent(nifti_files[idx + 1])
                input_2_enr = get_sub_image_from_file_path(
                    enr_2_nifti_file, mask_image, mask_indices)
                input_2 = get_array_union(input_2, input_2_enr)

            summed_array = input_1 + input_2

    if summed_array == None:
        summed_array = torch.zeros((182, 218, 182), dtype=torch.float32)

    return summed_array

# Convert sub-image array to original NIFTI dimensions based on mask indices
def convert_to_original_dimensions(origin_shape, sub_image_array, mask_indices):
    converted_array = np.zeros(origin_shape)
    converted_array[
        mask_indices[2][0]:mask_indices[2][1],
        mask_indices[1][0]:mask_indices[1][1],
        mask_indices[0][0]:mask_indices[0][1]
    ] = sub_image_array
    return converted_array

# Get image as MHA volume
def get_image_as_volume(image):
    uid = uuid.uuid4()
    mha_path = str(uid).replace('-', '') + '.mha'
    tmp_label_volume = os.path.join('/tmp/', mha_path)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(tmp_label_volume)
    writer.SetUseCompression(True)
    writer.Execute(image)

    with open(tmp_label_volume, 'rb') as f:
        vol_string = base64.encodestring(f.read()).decode('utf-8')

    os.remove(tmp_label_volume)

    return vol_string

# Save MHA volume as image
def save_volume_as_image(img_volume):
    uid = uuid.uuid4()
    uid_path_safe = str(uid).replace('-', '')
    mha_path = uid_path_safe + '.mha'
    tmp_label_volume = os.path.join('/tmp/', mha_path)

    volume_content = base64.b64decode(img_volume)

    with open(tmp_label_volume, 'wb') as w:
        w.write(volume_content)

    img = sitk.ReadImage(tmp_label_volume)
    img_path = '/tmp/' + uid_path_safe + '_input_image.nii'
    sitk.WriteImage(img, img_path)

    os.remove(tmp_label_volume)

    return img_path

# Get filter data and resulting image volumes
def get_filter_results(mask_path, filter_criteria):
    global torch_dict
    logging.info('Starting get_filter_results')
    mask = get_image_array(mask_path)
    mask_image = sitk.ReadImage(mask_path)

    # Get mask indices and convert them to numba list
    mask_indices = None  # get_mask_indices(mask)
    numba_mask_indices = None  # datatype_helper.convert_to_numba(mask_indices)

    # Calculate distance matrix for mask
    logging.info('Calculating distances')
    distance_matrix = get_or_create_distance_matrix(mask, torch_dict, mask_path)
    logging.info('Get_filtered_dataset')

    # Perform aggregation
    logging.info(f'n_filtered patients: {filter_criteria}')
    data_set, data_frame, data_frame_filtered = get_filtered_dataset(
        distance_matrix, filter_criteria)
    filtered_patient_amount = len(data_set['segmentation'])
    logging.info(f'n_filtered patients: {filtered_patient_amount}')
    logging.info('Aggregating results')
    aggregate_results = aggregate_dataset(data_set, mask, mask_image, numba_mask_indices)
    logging.info('Get Clin vars')
    clinical_variables = get_clinical_variables(
        data_frame, data_frame_filtered, get_supported_filter_criteria())
    logging.info('Aggregate results to CPU')

    # Resize the probability_map and sum_tumors_map to original dimensions
    resized_probability_map = aggregate_results['probability_map'].cpu().numpy()
    resized_sum_tumors_map = aggregate_results['sum_tumors'].cpu().numpy()

    logging.info('Storing results to file')
    # Get images from arrays and set origin, spacing, and direction
    sum_tumors_image = sitk.GetImageFromArray(resized_sum_tumors_map)
    probability_image = sitk.GetImageFromArray(resized_probability_map)

    sum_tumors_image.SetOrigin(mask_image.GetOrigin())
    sum_tumors_image.SetSpacing(mask_image.GetSpacing())
    sum_tumors_image.SetDirection(mask_image.GetDirection())

    probability_image.SetOrigin(mask_image.GetOrigin())
    probability_image.SetSpacing(mask_image.GetSpacing())
    probability_image.SetDirection(mask_image.GetDirection())

    # Convert to base64 encoding for JSON output
    based_64_encoded_probability_map = get_image_as_volume(probability_image)
    based_64_encoded_sum_tumors_map = get_image_as_volume(sum_tumors_image)

    # Format result dict
    result = {
        'filtered_patient_amount': filtered_patient_amount,
        'supported_filter_criteria': get_supported_filter_criteria(),
        'clinical_variables': clinical_variables,
        'probability_map': based_64_encoded_probability_map,
        'sum_tumors_map': based_64_encoded_sum_tumors_map
    }

    return result

# Function to get dataset based on filter criteria and distance matrix
def get_dataset(filter_criteria={}, distance_matrix=None):
    global torch_dict
    # Create virtual mask
    mask = np.ones((182, 218, 182))
    example_path = "/data/MNI152_T1_1mm.nii.gz"
    example_image = sitk.ReadImage(example_path)

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.SetOrigin(example_image.GetOrigin())
    mask_image.SetSpacing(example_image.GetSpacing())
    mask_image.SetDirection(example_image.GetDirection())
    sitk.WriteImage(mask_image, os.path.abspath('./whole-dataset.nii'))

    # Get mask indices and convert them to numba list
    mask_indices = None  # get_mask_indices(mask)
    numba_mask_indices = None  # datatype_helper.convert_to_numba(mask_indices)

    # Calculate distance matrix for mask
    distance_matrix = get_or_create_distance_matrix(mask, torch_dict, os.path.abspath('./whole-dataset.nii'))

    # Perform aggregation
    data_set, data_frame, data_frame_filtered = get_filtered_dataset(
        distance_matrix, filter_criteria)
    filtered_patient_amount = len(data_set['segmentation'])

    aggregate_results = aggregate_dataset(data_set, mask, mask_image, numba_mask_indices)
    clinical_variables = get_clinical_variables(
        data_frame, data_frame_filtered, get_supported_filter_criteria())

    # Resize the probability_map and sum_tumors_map to original dimensions
    resized_probability_map = aggregate_results['probability_map'].cpu().numpy()
    resized_sum_tumors_map = aggregate_results['sum_tumors'].cpu().numpy()

    # Get images from arrays and set origin, spacing, and direction
    sum_tumors_image = sitk.GetImageFromArray(resized_sum_tumors_map)
    probability_image = sitk.GetImageFromArray(resized_probability_map)

    sum_tumors_image.SetOrigin(mask_image.GetOrigin())
    sum_tumors_image.SetSpacing(mask_image.GetSpacing())
    sum_tumors_image.SetDirection(mask_image.GetDirection())

    probability_image.SetOrigin(mask_image.GetOrigin())
    probability_image.SetSpacing(mask_image.GetSpacing())
    probability_image.SetDirection(mask_image.GetDirection())

    # Convert to base64 encoding for JSON output
    based_64_encoded_probability_map = get_image_as_volume(probability_image)
    based_64_encoded_sum_tumors_map = get_image_as_volume(sum_tumors_image)

    # Format result dict
    result = {
        'filtered_patient_amount': filtered_patient_amount,
        'supported_filter_criteria': get_supported_filter_criteria(),
        'clinical_variables': clinical_variables,
        'probability_map': based_64_encoded_probability_map,
        'sum_tumors_map': based_64_encoded_sum_tumors_map
    }

    return result
