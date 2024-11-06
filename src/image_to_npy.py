#!/usr/bin/env python3
import os
import glob
import SimpleITK as sitk
import numpy as np


# Check if Numpy file is present for image and otherwise create it
def convert_image_to_npy(image_path):

    if 'nii' in image_path:
        npy_path = image_path.replace('nii', 'npy')

    if npy_path.endswith('.gz'):
        npy_path = npy_path.replace('.gz', '')

    if not os.path.isfile(npy_path):
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)

        with open(npy_path, 'wb') as f:
            np.save(f, image_array)


# Create binary Numpy files for batch of images
def batch_convert_images():

    dataset_dir = os.environ.get('DATASET_DIR', '/data/')

    image_list = glob.glob(dataset_dir + "*.nii*")

    for i in image_list:
        convert_image_to_npy(i)
