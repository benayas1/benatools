import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    v = tf.train.FloatList(value=value)
    return tf.train.Feature(float_list=v)

def _int64_feature(value):
    """Returns a single element int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    v = tf.train.Int64List(value=value)
    return tf.train.Feature(int64_list=v)

def serialize_example(data, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
      'image': _bytes_feature(data.tobytes()),  # img file to bytes
      'y':  _int64_feature(label),  # target
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

    # Create folder
    path = os.path.join(folder)
    if not os.path.exists(path):
        os.mkdir(path)

    # Create a dataframe of files and file sizes
    file_sizes = [avg_input_filesize] * len(file_list) if avg_input_filesize is not None else [os.path.getsize(f) for f
                                                                                               in file_list]
    df_files = pd.DataFrame({'path': file_list, 'label': labels, 'size': file_sizes})

    max_output_filesize = max_output_filesize * 1024 * 1024  # max file size in bytes

    df_files['file_id'] = (df_files['size'].cumsum() // max_output_filesize).astype(str)
    df_files['file_id'] = df_files['file_id'].str.zfill(zfill)

    path_prefix = folder + '/' + file_prefix
    for file_id, g in tqdm(df_files.groupby('file_id'), disable=not verbose):
        _write_tfrec_file(path_prefix + file_id, g['path'], g['label'], serialize_fn=serialize_fn, loader_fn=loader_fn,
                          dtype=dtype)


def convert_tfrecords(file_list,
                      labels,
                      folder='tfrecords',
                      file_prefix='file_',
                      max_output_filesize=200,
                      zfill=4,
                      dtype=np.int8,
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
        zfill: zero fill for the file name
        dtype: data type to convert to
        serialize_fn: Function to convert a single file into a tf.Example. Signature is (data, label)
        loader_fn: Function to load a file. Could be np.load if format is npy
        verbose
    """

    assert len(file_list) == len(labels)

    # Create folder
    path = os.path.join(folder)
    if not os.path.exists(path):
        os.mkdir(path)

    n_bytes = np.inf
    n_files = 0
    file_index = -1
    path = folder + '/' + file_prefix
    tmp = path + 'tmp.tfrec'

    for f, label in list(zip(file_list, labels)):
        data = loader_fn(f).astype(dtype).dtype(dtype)
        example = serialize_fn(data, label)

        n_bytes += data.nbytes

        if n_bytes > max_output_filesize * 1024 * 1024:
            # rename previous file
            if os.path.exists(tmp):
                filename = path + str(file_index).zfill(zfill) + '_' + str(n_files)
                os.rename(tmp, filename)
                if verbose:
                    print(f'File saved to {filename}')

            # Initialize next file
            n_files = 0
            n_bytes = data.nbytes
            file_index += 1
            with tf.io.TFRecordWriter(tmp) as writer:
                writer.write(example)
        else:
            with tf.io.TFRecordWriter(tmp, 'a') as writer:
                writer.write(example)

        n_files += 1

    # close last file
    if os.path.exists(tmp):
        filename = path + str(file_index).zfill(zfill) + '_' + str(n_files)
        os.rename(tmp, filename)
        if verbose:
            print(f'File saved to {filename}')
