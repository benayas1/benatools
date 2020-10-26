
# from https://github.com/qubvel/efficientnet/tree/master/efficientnet
import tensorflow as tf
import efficientnet.tfkeras as efn
import warnings

if tf.__version__ >= '2.3.0':
    warnings.warn('Official Implementation of Efficient Net is available from TF 2.3.0')

_EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

def create_efn(b, dim=128, weights='imagenet', include_top=False):
    """
    Provides an EfficientNet model. Implementation comes from
    https://github.com/qubvel/efficientnet/tree/master/efficientnet

    Parameters
    ----------
    b : int
        The EfficientNet version
    dim : int, Optional
        Input dimension, defaulted to 128
    weights : str
        pretrained weights, 'imagenet' or None
    include_top : bool
        Whether to include the network top or not. Defaults to False

    Returns
    -------
    tf.nn.Model
        The EfficientNet model
    """
    base = _EFNS[b](input_shape=(dim, dim, 3), weights=weights, include_top=include_top)
    return base
