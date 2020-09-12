import tensorflow as tf
import numpy as np
import os
import pandas as pd

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

def serialize_example(image, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
      'image': _bytes_feature(image),  # img file
      'y':  _int64_feature(label),  # target
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _write_tfrec_file(path, file_list, labels, serialize_fn, loader_fn=np.load, dtype=np.int8):
    assert len(file_list) == len(labels)

    with tf.io.TFRecordWriter(path + '.tfrec') as writer:
        for d in zip([file_list, labels]):
            data = loader_fn(d[0])
            label = d[1]
            example = serialize_fn(data.tobytes(), label)
            writer.write(example)


def convert(file_list,
            labels,
            folder='',
            file_prefix='file_',
            max_output_filesize=200,
            dtype=np.int8,
            avg_input_filesize=None,
            zfill=4,
            serialize_fn=serialize_example,
            loader_fn=np.load,
            ):

    # Create a dataframe of files and file sizes
    file_sizes = [avg_input_filesize] * len(file_list) if avg_input_filesize is None else [ os.path.getsize(f) for f in file_list]
    df_files = pd.DataFrame({'path':file_list, 'label':labels, 'size':file_sizes})

    max_output_filesize = max_output_filesize * 1024 * 1024  # max file size in bytes

    df_files['file_id'] = df_files['size'].cumsum() % max_output_filesize
    df_files['file_id'] = df_files['file_id'].str.zfill(zfill)

    path_prefix = folder + '/' + file_prefix
    for file_id, g in df_files.groupby('file_id'):
        _write_tfrec_file(path_prefix+file_id, df_files['path'], df_files['label'], serialize_fn=serialize_fn, loader_fn=loader_fn, dtype=dtype)

