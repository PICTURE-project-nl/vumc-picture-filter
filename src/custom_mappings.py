#!/usr/bin/env python3
import datetime
import math
import numpy as np
from dateutil.relativedelta import relativedelta
from datatype_helper import cast_to_digit, is_nan_digit
from datatype_helper import convert_options_and_binned_values
from collections import Counter


# Get mapping from API filter criteria to dataset columns
def get_filter_criteria_mapping(inverted=False):

    filter_criteria_mapping = {
        'Age': 'age',
        'KPS': 'kpspre',
        'Surgery': 'surgeryextend',
        'TumorStage': 'grade',
        
    }

    if inverted:
        filter_criteria_mapping = {
            v: k for k, v in filter_criteria_mapping.items()
        }

    return filter_criteria_mapping


# Get dataset column names which are binned in the UI
def get_binned_columns(inverted=False):
    binned_columns = [
            'age', 'birthyear', '5alascore', 'kpspre']

    if inverted:
        binned_columns = [v for k, v in \
            get_filter_criteria_mapping(True).items() if k in binned_columns]

    return binned_columns


# Get bin settings for column
def get_bin_settings(column):
    binned_column = False
    increment = None
    start = 0
    upper_boundary = None

    criteria_1_to_5_scale = ['5alascore']
    criteria_1_to_10_scale = ['kpspre', 'kpspos', 'kpsfol']

    if column in criteria_1_to_5_scale:
        increment = 1
        upper_boundary = 5

    if column in criteria_1_to_10_scale:
        increment = 1
        upper_boundary = 10

    if column == 'age':
        increment = 5
        upper_boundary = 100

    elif column == 'birthyear':
        increment = 5
        current_year = int(datetime.datetime.now().year)
        start = current_year - 100
        upper_boundary = current_year

    if increment and upper_boundary:
        binned_column = True

    return (binned_column, increment, start, upper_boundary)


# Generate bins with range min and max and optional increment
def generate_bins(min, max, increment=1):
    bins = []

    counter = 0
    for i, idx in enumerate(range(min, max)):
        if counter == 0:
            bins.append(i)

        counter+=1

        if counter == increment:
            counter = 0

    if min > 0:
        bins = [bin + min for bin in bins]

    return bins


# Add birthyear to data_frame
def add_birthyear(df):

    anon_id_column = 'AnonID'

    if 'anonID' in df.columns:
        anon_id_column = 'anonID'

    if 'age' in df.columns and not 'birthyear' in df.columns:

        df['birthyear'] = None

        anon_ids = df[anon_id_column].tolist()

        current_datetime = datetime.datetime.now()

        for anon_id in anon_ids:

            age = df.loc[df[anon_id_column] == anon_id, 'age'] .tolist()[0]

            if not is_nan_digit(age):

                age = cast_to_digit(age)
                years = int(math.floor(age))
                months = age - years

                birthyear = current_datetime - relativedelta(years=years)

                if months > 0:

                    months = int(months * 10)
                    months = int(months * 12/10)
                    birthyear = birthyear - relativedelta(months=months)

                df.loc[df[anon_id_column] == anon_id, 'birthyear'] = int(
                    birthyear.year)

    return df


# Get bin for input_val
def get_bin(bins, input_val, increment=1):

    input_val = int(input_val)
    result = None
    bins.sort()

    for idx, bin in enumerate(bins):

        bin_upper = bin + increment

        if input_val < bin_upper:
            result = bin_upper - increment
            return result
        else:
            if idx + 1 == len(bins) and input_val <= bin_upper:
                result = bin_upper - increment
                return result

    if not result:
        print("input_val out of boundaries")
        return result


# Get bins for list of values
def get_binned_values(options, values, increment=1):

    binned_values = {}

    for value in values:

        if isinstance(value, str):
            bin = value
        else:
            bin = get_bin(options, value, increment)

        if not bin:
            bin = value

        if bin not in binned_values.keys():
            binned_values[bin] = 1
        else:
            binned_values[bin] += 1

    for option in options:
        if option not in binned_values.keys():
            binned_values[option] = 0

    return binned_values


# Get values within filter_val range with optional max
def get_values_in_bin_range(column, values, min, max=None, include_unknown=False):

    values = values.tolist()

    binned_column, increment, start, upper_boundary = get_bin_settings(column)
    bins = generate_bins(start, upper_boundary, increment)
    min_input_bin = get_bin(bins, min, increment)

    values_in_bin_range = []

    if max:

        max_input_bin = get_bin(bins, max, increment)

    for i in values:

        if np.isnan(i):
            if include_unknown == True:
                values_in_bin_range.append(i)
                continue
            else:
                continue

        value_bin =  get_bin(bins, i, increment)

        if max and value_bin:
            if value_bin >= min_input_bin and value_bin <= max_input_bin:
                values_in_bin_range.append(i)
        elif value_bin:
            if value_bin == min_input_bin:
                values_in_bin_range.append(i)

    return values_in_bin_range


# Get options and values binned in column specific format
def get_binned_data(values, column):

    binned_column, increment, start, upper_boundary = get_bin_settings(column)
    options = []

    if binned_column:
        options = generate_bins(start, upper_boundary, increment)
        values = get_binned_values(options, values, increment)

        if column == 'age':
            values = sort_values(options, values)

        options, values = convert_options_and_binned_values(options, values)

    if not options:
        options = values

    options = {elem: elem for elem in options}

    return (options, values, upper_boundary)


# Get column options and values
def get_column_options_and_values(
    data_type, values, filter_criteria_mapping, column):

    upper_boundary = None
    options = {}
    values, value_counts, nan_replacement = get_category_counts(values, data_type)
    binned_columns = get_binned_columns()

    boolean_options = {
        True: "Yes",
        False: "No"
    }

    if data_type == "boolean":
        options = boolean_options

        values_sorted = {}

        for option in options:

            if option in value_counts.keys():
                values_sorted[option] = value_counts[option]
            else:
                values_sorted[option] = 0

        values = values_sorted

    elif data_type == "string":
        options = {elem: elem for elem in list(value_counts.keys())}

        validation_values = [True, False]
        filtered_values = [elem for elem in values if elem != nan_replacement]

        if all(elem in validation_values for elem in filtered_values):

            options = boolean_options
            options[nan_replacement] = nan_replacement
            value_counts = values = sort_values(options, value_counts)

        values = value_counts

    elif data_type == "float" or data_type == "integer":

        if column in binned_columns:
            options, values, upper_boundary = get_binned_data(values, column)
        else:
            values = value_counts
            options = {elem: elem for elem in list(value_counts.keys())}

    return (options, values, data_type, upper_boundary)


# Get counts for categories in values
def get_category_counts(values, data_type, nan_handling='replace'):

    nan_replacement = ''

    if nan_handling == 'replace':
        nan_replacement = 'Unknown'

        values.replace({np.nan: nan_replacement}, inplace=True)
        values = values.tolist()

    elif nan_handling == 'remove':

        values = values.dropna().tolist()

    value_counts = Counter(values)

    return values, value_counts, nan_replacement


# Return values sorted in order of the options
def sort_values(options, values):

    values_sorted = {}

    for option in options:
        values_sorted[option] = values[option]

    return values_sorted
