#!/usr/bin/env python3
import numpy as np
from numba.typed import List
from math import isnan


# Cast input_val to boolean in chosen representation
def cast_to_boolean(input_val, string_representation=False):

    boolean_true_values = ['true', 'True', True, 'TRUE', 1, '1', 'yes', 'Yes']
    boolean_false_values = ['false', 'False', False, 'FALSE', 0, '0', 'no', 'No']

    if input_val in boolean_true_values:
        input_val = True

    elif input_val in boolean_false_values:
        input_val = False

    if string_representation:

        if input_val == True:
            input_val = 'True'
        elif input_val == False:
            input_val = 'False'

    return input_val


# Cast list or dict with floats to integers
def floats_to_integers(input_values):

    casted_values = None

    if isinstance(input_values, list):

        casted_values = []

        for i in input_values:
            if isinstance(i, float):
                i = int(i)

            casted_values.append(i)

        casted_values.sort()

    if isinstance(input_values, dict):

        casted_values = {}

        for k, v in input_values.items():
            if isinstance(k, float):
                k = int(k)

            casted_values[k] = v

        unknown_entry = None

        if "Unknown" in casted_values:
            unknown_entry = casted_values["Unknown"]
            casted_values.pop("Unknown")

        casted_values = dict(sorted(casted_values.items()))

        if unknown_entry:
            casted_values["Unknown"] = unknown_entry

    if not casted_values:

        # Unsupported datatype, returning input_values
        casted_values = input_values

    return casted_values


# Return tuple of options and binned values converted to integers
def convert_options_and_binned_values(options, values):

    options = floats_to_integers(options)
    values = floats_to_integers(values)

    return (options, values)


# Check if input_val can be cast to float
def is_float(input_val):

    if isinstance(input_val, str):
        if input_val.isalpha():
            return False
    try:
        float(input_val)
        return True
    except ValueError:
        return False


# Check if input_val is a NaN digit
def is_nan_digit(input_val):

    digit = cast_to_digit(input_val)

    if digit:
        if isnan(digit):
            return True

    return False


# Cast input_val to digit or return False is not a valid digit
def cast_to_digit(input_val, round_to_int=False):

    if is_float(input_val):
        input_val = float(input_val)
    else:
        return False

    if round_to_int:
        input_val = round(input_val)

    return input_val


# Convert native datatype to Numba equivalent
def convert_to_numba(native_var):

    numba_var = None

    if type(native_var) == list:
        numba_var = List()

        for i in native_var:
            numba_var.append(i)

    return numba_var


# Get mapping of np data_types to client representation
def get_data_type_mapping():

    data_type_mapping = {
        np.int64: 'integer',
        np.float64: 'float',
        object: 'string',
        bool: 'boolean'
    }

    return data_type_mapping
