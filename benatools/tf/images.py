import tensorflow as tf
import math
import tensorflow.keras.backend as K
import numpy as np
import collections

def _check_rotation_arg(x):
    """ Returns a list of rotation args"""
    if x is None:
        return [90, 90, 90]

    if np.isscalar(x):
        return [x, x, x]

    if isinstance(x, (collections.Sequence, np.ndarray, tf.Tensor)):
        if len(x) < 3:
            raise Exception("Rotation parameter must have length 3")
        return x[:3]

    raise Exception("Rotation parameter must be a scalar or a list of length 3")


def get_mat3d(rotation=None, shear=2.0, x_zoom=8.0, y_zoom=8.0, z_zoom=8.0, x_shift=8.0, y_shift=8.0, z_shift=8.0):
    """
    Creates a transformation matrix which rotates, shears, zooms and shift an 2D image.

    Parameters
    ----------
    rotation : float
        Degrees to rotate
    shear : float
        Degrees to shear
    height_zoom : float
        height zoom ratio
    width_zoom : float
        width zoom ratio
    height_shift : float
        height shift ratio
    width_shift : float
        width shift ratio

    Returns
    -------
    tf.tensor
        3x3 transformation matrix
    """

    # CONVERT DEGREES TO RADIANS
    rotation = _check_rotation_arg(rotation)

    def get_4x4_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [4, 4])

    # ROTATION MATRIX
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    # X axis
    r = math.pi * rotation[0] / 180.
    cx = tf.math.cos(r)
    sx = tf.math.sin(r)
    rx = get_4x4_mat([one,  zero, zero, zero,
                      zero, cx, -sx, zero,
                      zero, sx,  cx, zero,
                      zero, zero, zero, one])

    # Y axis
    r = math.pi * rotation[1] / 180.
    cy = tf.math.cos(r)
    sy = tf.math.sin(r)
    ry = get_4x4_mat([cy,  zero, sy, zero,
                      zero, one, zero, zero,
                      -sy, zero,  cy, zero,
                      zero, zero, zero, one])

    # Z axis
    r = math.pi * rotation[2] / 180.
    cz = tf.math.cos(r)
    sz = tf.math.sin(r)
    rz = get_4x4_mat([cz, -sz, zero, zero,
                      sz, cz, zero, zero,
                      zero, zero,  one, zero,
                      zero, zero, zero, one])

    rotation_matrix = np.random.choice([rx, ry, rz])

    # SHEAR MATRIX
    shear = math.pi * shear / 180.
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_4x4_mat([one, s2, zero,
                                zero, c2, zero,
                                zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_4x4_mat([one / x_zoom, zero, zero, zero,
                               zero, one / y_zoom, zero, zero,
                               zero, zero, one/ z_zoom, zero,
                               zero, zero, zero , one])
    # SHIFT MATRIX
    shift_matrix = get_4x4_mat([one, zero, zero, x_shift,
                                zero, one, zero, y_shift,
                                zero, zero, one, z_shift,
                                zero, zero, zero, one])

    return K.dot(rotation_matrix,
                 K.dot(zoom_matrix, shift_matrix))

def transform3d(object, dimension, rotation=None, shear=2.0, x_zoom=8.0, y_zoom=8.0, z_zoom=8.0, x_shift=8.0, y_shift=8.0, z_shift=8.0, prob=0.5):
    """
    Rotates, shears, zooms and shift an single object, not a batch of them.

    Parameters
    ----------
    image : tf.Tensor of shape [h,w,d,c]
        A single image to be transformed
    dimension : int
        Dimension in pixels of the squared image
    rotation : float
        Degrees to rotate
    shear : float
        Degrees to shear
    height_zoom : float
        height zoom ratio
    width_zoom : float
        width zoom ratio
    height_shift : float
        height shift ratio
    width_shift : float
        width shift ratio

    Returns
    -------
    tf.Tensor
        A transformed object
    """

    rotation = _check_rotation_arg(rotation)

    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if P == 0:
        return object  # no action

    DIM = dimension
    XDIM = DIM % 2

    rot = rotation * tf.random.normal([3], dtype='float32')
    shr = shear * tf.random.normal([1], dtype='float32')
    x_zoom = 1.0 + tf.random.normal([1], dtype='float32') / x_zoom
    y_zoom = 1.0 + tf.random.normal([1], dtype='float32') / y_zoom
    z_zoom = 1.0 + tf.random.normal([1], dtype='float32') / z_zoom
    x_shift = x_shift * tf.random.normal([1], dtype='float32')
    y_shift = y_shift * tf.random.normal([1], dtype='float32')
    z_shift = z_shift * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat3d(rot, shr, x_zoom, y_zoom, z_zoom, x_shift, y_shift, z_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(object, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, DIM, 3])

def get_mat2d(rotation=180.0, shear=2.0, height_zoom=8.0, width_zoom=8.0, height_shift=8.0, width_shift=8.0):
    """
    Creates a transformation matrix which rotates, shears, zooms and shift an 2D image.

    Parameters
    ----------
    rotation : float
        Degrees to rotate
    shear : float
        Degrees to shear
    height_zoom : float
        height zoom ratio
    width_zoom : float
        width zoom ratio
    height_shift : float
        height shift ratio
    width_shift : float
        width shift ratio

    Returns
    -------
    tf.tensor
        3x3 transformation matrix
    """

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_mat([c1, s1, zero,
                                   -s1, c1, zero,
                                   zero, zero, one])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one, s2, zero,
                                zero, c2, zero,
                                zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero,
                               zero, one / width_zoom, zero,
                               zero, zero, one])
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one, zero, height_shift,
                                zero, one, width_shift,
                                zero, zero, one])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix, shift_matrix))


def transform2d(image, dimension, rotate=180.0, shear=2.0, hzoom=8.0, vzoom=8.0, hshift=8.0, wshift=8.0, prob=0.5):
    """
    Rotates, shears, zooms and shift an single image, not a batch of them.

    Parameters
    ----------
    image : tf.Tensor of shape [h,w,c]
        A single image to be transformed
    dimension : int
        Dimension in pixels of the squared image
    rotation : float
        Degrees to rotate
    shear : float
        Degrees to shear
    height_zoom : float
        height zoom ratio
    width_zoom : float
        width zoom ratio
    height_shift : float
        height shift ratio
    width_shift : float
        width shift ratio

    Returns
    -------
    tf.Tensor
        A transformed image
    """

    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if P == 0:
        return image  # no action

    DIM = dimension
    XDIM = DIM % 2  # fix for size 331

    rot = rotate * tf.random.normal([1], dtype='float32')
    shr = shear * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / vzoom
    h_shift = hshift * tf.random.normal([1], dtype='float32')
    w_shift = wshift * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat2d(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def dropout(image, prob=0.75, ct=8, sz=0.2):
    """
    Coarse dropout randomly remove squares from training images

    Parameters
    ----------
    image : tf.Tensor
        image of size [height,width,3] not a batch of [b,dim,dim,3]
    prob : float
        probability to perform dropout
    ct : int
        number of squares to remove
    sz : size
        size of square (in % of the image dimension)

    Returns
    -------
        image with ct squares of side size sz*dimension removed
    """

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0) | (ct == 0) | (sz == 0):
        return image # no action

    h, w, c = image.shape

    sq_height = tf.cast(sz * h, tf.int32) * P
    sq_width = tf.cast(sz * w, tf.int32) * P

    # generate random black squares
    for k in range(ct):
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, w), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, h), tf.int32)
        # COMPUTE SQUARE
        ya = tf.math.maximum(0, y - sq_height // 2)
        yb = tf.math.minimum(h, y + sq_height // 2)
        xa = tf.math.maximum(0, x - sq_width // 2)
        xb = tf.math.minimum(w, x + sq_width // 2)
        # DROPOUT IMAGE
        one = image[ya:yb, 0:xa, :]
        two = tf.zeros([yb - ya, xb - xa, 3])
        three = image[ya:yb, xb:w, :]
        middle = tf.concat([one, two, three], axis=1)
        image = tf.concat([image[0:ya, :, :], middle, image[yb:h, :, :]], axis=0)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    image = tf.reshape(image, [h, w, 3])
    return image

def mixup_labels(label1, label2, n_classes, a):
    if len(label1.shape) == 1:
        lab1 = tf.one_hot(label1, n_classes)
        lab2 = tf.one_hot(label2, n_classes)
    else:
        lab1 = label1
        lab2 = label2
    return (1 - a) * lab1 + a * lab2


def cutmix(image, label, batch_size=16, dimension=256, n_classes = 1, prob=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0):
        return image  # no action

    DIM = dimension

    imgs = []
    labs = []
    for j in range(batch_size):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= prob, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
        WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        # MAKE CUTMIX IMAGE
        one = image[j, ya:yb, 0:xa, :]
        two = image[k, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
        if len(label.shape) == 1:
            lab1 = tf.one_hot(label[j], n_classes)
            lab2 = tf.one_hot(label[k], n_classes)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(batch_size, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs),(batch_size, n_classes))
    return image2,label2

def mixup(image, label, batch_size=16, dimension=256, n_classes=1, prob=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0):
        return image  # no action

    DIM = dimension
    CLASSES = n_classes

    imgs = []
    labs = []
    for j in range(batch_size):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= prob, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1 - a) * img1 + a * img2)
        # MAKE CUTMIX LABEL
        if len(label.shape) == 1:
            lab1 = tf.one_hot(label[j], CLASSES)
            lab2 = tf.one_hot(label[k], CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (batch_size, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (batch_size, CLASSES))
    return image2, label2


def spec_augmentation(image, prob=0.66, time_drop_width=0.0625, time_stripes_num=2, freq_drop_width=0.125,
                      freq_stripes_num=2, height=64, width=501):
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0):
        return image  # no action

    time_drop_size = tf.cast(time_drop_width * width, tf.int32)

    for i in range(time_stripes_num):
        begin = tf.cast(tf.random.uniform([], 0, width), tf.int32)
        end = tf.cast(tf.math.minimum(width, begin + time_drop_size), tf.int32)
        zeros = tf.zeros([height, end - begin, 3])
        image = tf.concat([image[:, :begin, :], zeros, image[:, end:, :]], axis=1)

    freq_drop_size = tf.cast(freq_drop_width * height, tf.int32)

    for i in range(freq_stripes_num):
        begin = tf.cast(tf.random.uniform([], 0, height), tf.int32)
        end = tf.cast(tf.math.minimum(height, begin + freq_drop_size), tf.int32)
        zeros = tf.zeros([end - begin, width, 3])
        image = tf.concat([image[:begin, :, :], zeros, image[end:, :, :]], axis=0)

    image = tf.reshape(image, [height, width, 3])

    return image


def add_white_noise(image, prob=0.3, std=0.2):
    """
    Add white noise to object or image

    Parameters
    ----------
    image : tf.Tensor
        image of size [height,width,3] not a batch of [b,dim,dim,3]
    prob : float
        probability to perform dropout
    std : size
        Number of standard deviations to calculate noise with

    Returns
    -------
    tf.Tensor
        input image or object with added white noise
    """
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0):
        return image  # no action

    h, w, c = image.shape

    noise = tf.random.normal(shape=image.shape, mean=tf.reduce_mean(image), stddev=tf.math.reduce_std(image) * std)
    image = image + noise
    image = tf.reshape(image, image.shape)
    return image


def add_band_noise(image, prob=0.3, std=0.2, band_height=0.125):
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0):
        return image  # no action

    h, w, c = image.shape

    band_height = tf.cast(band_height * h, tf.int32)

    begin = tf.cast(tf.random.uniform([], 0, h), tf.int32)
    end = tf.cast(tf.math.minimum(h, begin + band_height), tf.int32)

    noise = tf.random.normal(shape=[end - begin, w, 3], mean=tf.reduce_mean(image),
                             stddev=tf.math.reduce_std(image) * std)
    image = tf.concat([image[:begin, :, :], image[begin:end, :, :] + noise, image[end:, :, :]], axis=0)
    image = tf.reshape(image, [h, w, c])
    return image
