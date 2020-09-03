import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


def get_lr_callback(lr_start=0.000005, lr_max=0.00000125, lr_min=0.000001, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8):
    """ Train schedule for transfer learning. The learning rate starts near zero, then increases to a maximum, then decays over time.
        A good practice to follow is to increase maximum learning rate as batch size increase

        Input:
            batch_size
            lr_start: initial learning rate value
            lr_max: maximum learning rate.
            lr_min: minimum learning rate.
            lr_ramp_ep: number of epochs of ramp up
            lr_sus_ep: number of epochs of plateau
            lr_decay: decay [0,1]
    """

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback



def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def binary_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.mean(flag, axis=-1)

class F1Callback(Callback):
    def __init__(self):
        Callback.__init__()
        self.f1s = []

    def on_epoch_end(self, epoch, logs=None):
        eps = np.finfo(np.float32).eps
        recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)
        precision = logs["val_true_positives"] / (logs["val_predicted_positives"] + eps)
        f1 = 2*precision*recall / (precision+recall+eps)
        print("f1_val (from log) =", f1)
        self.f1s.append(f1)

def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))

def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))

def F1(y_true, y_pred):
    TPFN = possible_positives(y_true, y_pred)
    TPFP = predicted_positives(y_true, y_pred)
    TP = true_positives(y_true, y_pred)
    return (TP * 2) / (TPFN + TPFP + K.epsilon())