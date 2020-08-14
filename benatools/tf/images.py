import tensorflow as tf
import math
import tensorflow.keras.backend as K


def get_mat(rotation=180.0, shear=2.0, height_zoom=8.0, width_zoom=8.0, height_shift=8.0, width_shift=8.0):
    """Creates a transformation matrix which rotates, shears, zooms and shift an image.

    Inputs:
        rotation: degrees to rotate
        shear: degrees to shear
        height_zoom: height zoom ratio
        width_zoom: width zoom ratio
        height_shift: height shift ratio
        width_shift: width shift ratio

    Outputs:
        3 x3 transformation matrix
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


def transform(image, dimension, rotate=180.0, shear=2.0, hzoom=8.0, vzoom=8.0, hshift=8.0, wshift=8.0):
    """ transforms an image

    Input:
        image: image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        dimension: image dimension
        rotation: degrees to rotate
        shear: degrees to shear
        height_zoom: height zoom ratio
        width_zoom: width zoom ratio
        height_shift: height shift ratio
        width_shift: width shift ratio

    Output:
        image randomly rotated, sheared, zoomed, and shifted
    """

    DIM = dimension
    XDIM = DIM % 2  # fix for size 331

    rot = rotate * tf.random.normal([1], dtype='float32')
    shr = shear * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / vzoom
    h_shift = hshift * tf.random.normal([1], dtype='float32')
    w_shift = wshift * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

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


def dropout(image, dimension=256, prob=0.75, ct=8, sz=0.2):
    """ Coarse dropout randomly remove squares from training images

    Input:
        image: image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        dimension: image dimension
        prob: probability to perform dropout
        ct: number of squares to remove
        sz: size of square (in % of the image dimension)

    Output:
        image with ct squares of side size sz*dimension removed
    """
    DIM = dimension

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0) | (ct == 0) | (sz == 0): return image # no action

    # generate random black squares
    for k in range(ct):
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        # COMPUTE SQUARE
        WIDTH = tf.cast(sz * DIM, tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        # DROPOUT IMAGE
        one = image[ya:yb, 0:xa, :]
        two = tf.zeros([yb - ya, xb - xa, 3])
        three = image[ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        image = tf.concat([image[0:ya, :, :], middle, image[yb:DIM, :, :]], axis=0)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    image = tf.reshape(image, [DIM, DIM, 3])
    return image