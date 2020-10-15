import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def float_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    v = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list=v)

def int64_feature(value):
    """Returns a single element int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    v = tf.train.Int64List(value=value)
    return tf.train.Feature(int64_list=v)

def serialize_example(data, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
      'x': bytes_feature(data.tobytes()),  # x file to bytes
      'y': int64_feature(label),  # target
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _write_tfrec_file(path, file_list, labels, serialize_fn, loader_fn=np.load, dtype=np.int8):
    """
    Converts a list of files to TfRecords, and then writes them into a tfrec file

    Inputs:
        path: The destination path for the tfrec file
        file_list: A list of paths to be encoded into the tfrec file
        labels: A list of labels
        serialize_fn: Function to convert a single file into a tf.Example. Signature is (data, label)
        loader_fn: Function to load a file. Could be np.load if format is npy
        dtype: Data type to convert loaded data before serializing. For images, np.int8
    """
    assert len(file_list) == len(labels)

    with tf.io.TFRecordWriter(path + '.tfrec') as writer:
        for d in list(zip(file_list, labels)):
            data = loader_fn(d[0]).astype(dtype)
            label = d[1]
            example = serialize_fn(data, label)
            writer.write(example)


def _write_tfrec_file2(file_list,
                       labels,
                       folder,
                       file_index,
                       serialize_fn,
                       max_bytes=200*1024*1024,
                       loader_fn=np.load,
                       file_prefix='',
                       dtype=np.int8,
                       zfill=5,
                       verbose=False):
    """
    Converts a list of files to TfRecords, and then writes them into a tfrec file.
    The file name will be: <folder>/<file_prefix><file_index>_<n_files>.tfrec

    Inputs:
        file_list: A list of paths to be encoded into the tfrec file
        labels: A list of labels
        folder: The destination folder for the tfrec file
        file_index: The index of the file within the folder
        serialize_fn: Function to convert a single file into a tf.Example. Signature is (data, label)
        max_bytes: maximum size in bytes. Default to 200 MB
        loader_fn: Function to load a file. Could be np.load if format is npy
        dtype: Data type to convert loaded data before serializing. For images, np.int8
        zfill:
        verbose: True to print filename
    """
    assert len(file_list) == len(labels); "file_list and label_list must be the same length"

    n_bytes, n_files = 0, 0
    tmp = folder + '/tmp.tfrec'
    with tf.io.TFRecordWriter(tmp) as writer:
        for d in list(zip(file_list, labels)):
            data = loader_fn(d[0]).astype(dtype)
            label = d[1]
            example = serialize_fn(data, label)
            writer.write(example)

            n_bytes += data.nbytes
            n_files += 1

            # break execution if files surpasses max number of bytes
            if n_bytes > max_bytes:
                break

    # rename tmp file
    if os.path.exists(tmp):
        filename = folder + '/' + file_prefix + str(file_index).zfill(zfill) + '_' + str(n_files) + '.tfrec'
        os.rename(tmp, filename)
        if verbose:
            print(f'File saved to {filename}')

    return n_files


def convert(file_list,
            labels,
            folder='tfrecords',
            file_prefix='file_',
            max_output_filesize=200,
            dtype=np.int8,
            avg_input_filesize=None,
            zfill=4,
            serialize_fn=serialize_example,
            loader_fn=np.load,
            verbose=True
            ):
    """
    Converts a list of files into a set of tfrec files

    Inputs:
        file_list: A list of paths to be encoded into the tfrec file
        labels: A list of labels
        folder: Folder where all the tfrec files will be stored
        file_prefix: prefix to place in
        max_output_filesize: Maximum file size in MB
        dtype:
        avg_input_filesize:
        zfill:
        serialize_fn: Function to convert a single file into a tf.Example. Signature is (data, label)
        loader_fn: Function to load a file. Could be np.load if format is npy
        dtype: Data type to convert loaded data before serializing. For images, np.int8
    """
    assert len(file_list) == len(labels); "file_list and label_list must be the same length"

    # Create folder
    path = os.path.join(folder)
    if not os.path.exists(path):
        os.mkdir(path)

    # Create a dataframe of files and file sizes
    file_sizes = [avg_input_filesize] * len(file_list) if avg_input_filesize is not None else [os.path.getsize(f) for f
                                                                                               in file_list]
    df_files = pd.DataFrame({'path': file_list, 'label_idx': [i for i in range(len(labels))], 'size': file_sizes})

    max_output_filesize = max_output_filesize * 1024 * 1024  # max file size in bytes

    df_files['file_id'] = (df_files['size'].cumsum() // max_output_filesize).astype(str)
    df_files['file_id'] = df_files['file_id'].str.zfill(zfill)

    path_prefix = folder + '/' + file_prefix
    for file_id, g in tqdm(df_files.groupby('file_id'), disable=not verbose):
        _write_tfrec_file(path_prefix + file_id + '_' + str(len(g)).zfill(zfill),
                          g['path'],
                          labels[g['label_idx'].values],
                          serialize_fn=serialize_fn,
                          loader_fn=loader_fn,
                          dtype=dtype)


def convert2(file_list,
             label_list,
             folder='tfrecords',
             file_prefix='file_',
             max_output_filesize=200,
             dtype=np.int8,
             zfill=5,
             serialize_fn=serialize_example,
             loader_fn=np.load,
             verbose=True
             ):
    """
    Converts a list of files into a set of tfrec files

    Inputs:
        file_list: A list of paths to be encoded into the tfrec file
        labels: A list of labels
        folder: Folder where all the tfrec files will be stored
        file_prefix: prefix to place in
        max_output_filesize: Maximum file size in MB
        dtype:
        avg_input_filesize:
        zfill:
        serialize_fn: Function to convert a single file into a tf.Example. Signature is (data, label)
        loader_fn: Function to load a file. Could be np.load if format is npy
        dtype: Data type to convert loaded data before serializing. For images, np.int8
    """
    assert len(file_list) == len(label_list); "file_list and label_list must be the same length"

    # Create folder
    path = os.path.join(folder)
    if not os.path.exists(path):
        os.mkdir(path)

    max_output_filesize = max_output_filesize * 1024 * 1024  # max file size in bytes

    files = file_list.copy()
    labels = label_list.copy()
    file_index=0
    while True:
        n = _write_tfrec_file2(files,
                               labels,
                               folder=folder,
                               file_index=file_index,
                               file_prefix=file_prefix,
                               serialize_fn=serialize_fn,
                               max_bytes=max_output_filesize,
                               loader_fn=loader_fn,
                               dtype=dtype,
                               zfill=zfill,
                               verbose=verbose)
        file_index += 1
        files = files[n:]
        labels = labels[n:]


def read_labeled_tfrecord(ex):
    """
    This is an example of decoding a tf record. You should know before hand the tf record format, and
    define it in a dictionary.

    Inputs:
        ex: is an tf example object, provided by the TFRecordDataset
    Outputs:
        data: the decoded data
        label: the label of this example

    More parameters, inputs or outputs, can be added to this function.
    """
    labeled_tfrec_format = {
        'x': tf.io.FixedLenFeature([], tf.string),  # data
        'y': tf.io.FixedLenFeature([], tf.int64),  # label
    }
    example = tf.io.parse_single_example(ex, labeled_tfrec_format)
    x = tf.io.decode_raw(example['x'], out_type=tf.int8)
    x = tf.cast(x, tf.float32) / 255.0

    # Labels
    y = tf.cast(example['y'], tf.float32)

    return x, y