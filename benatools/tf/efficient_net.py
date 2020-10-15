
# from https://github.com/qubvel/efficientnet/tree/master/efficientnet
import tensorflow as tf
import efficientnet.tfkeras as efn
import warnings

if tf.__version__ >= '2.3.0':
    warnings.warn('Official Implementation of Efficient Net is available from TF 2.3.0')

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3,
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

def create_efn(b, dim=128, weights='imagenet', include_top=False):
    base = EFNS[b](input_shape=(dim, dim, 3), weights=weights, include_top=include_top)
    return base
