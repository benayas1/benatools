import tensorflow.keras.backend as K

def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def binary_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.mean(flag, axis=-1)

def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def false_negatives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * (1-y_pred), 0, 1)))

def true_negatives(y_true, y_pred):
    return K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))

def false_positives(y_true, y_pred):
    return K.sum(K.round(K.clip((1-y_true) * y_pred, 0, 1)))

def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))

def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))

def recall(y_true, y_pred):
    TP = true_positives(y_true, y_pred)
    FN = false_negatives(y_true, y_pred)
    return TP / (TP+FN)

def precission(y_true, y_pred):
    TP = true_positives(y_true, y_pred)
    FP = false_positives(y_true, y_pred)
    return TP / (TP+FP)

def F1(y_true, y_pred):
    TPFN = possible_positives(y_true, y_pred)
    TPFP = predicted_positives(y_true, y_pred)
    TP = true_positives(y_true, y_pred)
    return (TP * 2) / (TPFN + TPFP + K.epsilon())

def F1_2(y_true, y_pred):
    P = precission(y_true, y_pred)
    R = recall(y_true, y_pred)
    return (2 * P * R) / (P + R + K.epsilon())