# -*- coding: utf-8 -*-

# commented version of original from kaggle at
# https://www.kaggle.com/pavansanagapati/12-simple-tips-to-save-ram-memory-for-1-gb-dataset
from sys import getsizeof
import numpy as np
import pandas as pd
import gc
import subprocess
import os

import warnings

warnings.simplefilter("ignore")

INPUT_PATH = "/content"


def create_df():
    train = pd.read_csv(
        os.path.join(
            INPUT_PATH,
            "jigsaw-unintended-bias-train-processed-seqlen128.csv"))
    return train


train.head()

# using gc.collect(): the dataframe not being used can be deleted using del command and just to make sure there is no residual memory usage we can call gc.collect()
# calling gc.collect after transformations/functions also help to remove
# the accumulations
del train
gc.collect()

# using datatype conversion
# we can get the memory usage by calling the info on the dataframe
train = create_df()
train.info(memory_usage="deep")

# strings are stored as objects and mixed data types for other columns.
# each type is based on the specialized class in the pandas.core.internals module. Pandas uses the objectblock class to represent the block containing string columns and floatblocks to repsent the float columns.
# the numeric columns are stored as Numpy ndarray which is built around C array and the values are stored in contigous memory block which makes accessing values incredibly fast.
# memory usage for each data type::
select = train.select_dtypes(include=["int"])
print(select.memory_usage(deep=True))

for dtype in ["float", "int", "object"]:
    selected_dtype = train.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Memory usage for {} columns: {:03.2f} MB".format(dtype, mean_usage_mb))

# using subtypes like float16, float32 etc to reduce memory
for i in ["uint8", "int8", "int16"]:
    print(np.iinfo(i))


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)


train_int = train.select_dtypes(include=["int"])
converted_train_int = train_int.apply(pd.to_numeric, downcast="unsigned")
print(mem_usage(train_int))
print(mem_usage(converted_train_int))

compare_ints_table = pd.concat(
    [train_int.dtypes, converted_train_int.dtypes], axis=1)
compare_ints_table.columns = ["Before Conversion", "After Conversion"]
print(compare_ints_table)
compare_ints_table.apply(pd.Series.value_counts)

train_float = train.select_dtypes(include=["float"])
converted_floats = train_float.apply(pd.to_numeric, downcast="float")
print(mem_usage(train_float))
print(mem_usage(converted_floats))
compare_floats = pd.concat(
    [train_float.dtypes, converted_floats.dtypes], axis=1)
compare_floats.columns = ["Before", "After"]
compare_floats.apply(pd.Series.value_counts)

optimized_train = train.copy()
optimized_train[converted_train_int.columns] = converted_train_int
optimized_train[converted_floats.columns] = converted_floats
print(mem_usage(train))
print(mem_usage(optimized_train))

# python doesnot have fine grained-control over how memory is stored.
# hence, it's slower to access the strings.

for s in ["Hello", "Hello Hi", "Python Python hello", "Hi Hi HI"]:
    print(getsizeof(s))

# string's mem usage is the same as in python for pandas
obj_ser = pd.Series(["Hello", "Hello Hi", "Python Python hello", "Hi Hi HI"])
obj_ser.apply(getsizeof)

# object types can be optimized using categoricals by using int to represent values in column
# ints are then mapped to the original raw values using another dict
train_obj = train.select_dtypes(include=["object"]).copy()
train_obj.describe()

ratings_col = train.rating
print(ratings_col.head())
rating_cat = ratings_col.astype("category")
print(rating_cat.head())

# missing values would be assigned -1
rating_cat.head().cat.codes

print(mem_usage(ratings_col))
print(mem_usage(rating_cat))

# convert to category if less than 50% data is unique
train_new = pd.DataFrame()
for col in train.columns:
    num_unique_values = len(train[col].unique())
    total = len(train[col])
    if num_unique_values / total < 0.5:
        train_new.loc[:, col] = train[col].astype("category")
    else:
        train_new.loc[:, col] = train[col]

print(mem_usage(train))
print(mem_usage(train_new))

optimized_train[train_new.columns] = train_new
mem_usage(optimized_train)

# if we can't even load the dataframe
dtypes = optimized_train.dtypes
dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]
column_types = dict(zip(dtypes_col, dtypes_type))
print(column_types)

# using the dictionary we can read in the dataframe
read_optimized = pd.read_csv(
    os.path.join(
        INPUT_PATH,
        "jigsaw-unintended-bias-train-processed-seqlen128.csv"),
    dtype=column_types,
)
print(mem_usage(read_optimized))

# importing select number of rows
train_df = pd.read_csv("path", nrows=100)

# random row selection (#4)


def file_len(fname):
    process = subprocess.Popen(
        ["wc", "-l", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    result, error = process.communicate()
    if process.returncode != 0:
        raise IOError(error)
    return int(result.strip().split()[0])


random_rows = file_len(
    os.path.join(
        INPUT_PATH,
        "jigsaw-unintended-bias-train-processed-seqlen128.csv"))
print(random_rows)

skip_rows = np.random.choice(
    np.arange(1, random_rows), size=random_rows - 1 - 10000, replace=False
)

print(skip_rows)

# passing the skiprows while reading the csv file
# and del the skip_rows and gc.collect()
# skiping without the header skiprows=range(1,2000)

# using generators, itertools, not using "+" for strings
# using %memit similar to %timeit
# %memit function()

# reducing memory leaks using del function
# using line_profiler, memory_profiler,
