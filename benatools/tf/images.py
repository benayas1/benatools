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
    """ transforms an image by rotating, zooming a shearing

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


def dropout(image, height=256, width=256, prob=0.75, ct=8, sz=0.2):
    """ Coarse dropout randomly remove squares from training images

    Input:
        image: image of size [height,width,3] not a batch of [b,dim,dim,3]
        dimension: image dimension
        prob: probability to perform dropout
        ct: number of squares to remove
        sz: size of square (in % of the image dimension)

    Output:
        image with ct squares of side size sz*dimension removed
    """

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0) | (ct == 0) | (sz == 0):
        return image # no action

    sq_height = tf.cast(sz * height, tf.int32) * P
    sq_width = tf.cast(sz * width, tf.int32) * P

    # generate random black squares
    for k in range(ct):
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, width), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, height), tf.int32)
        # COMPUTE SQUARE
        ya = tf.math.maximum(0, y - sq_height // 2)
        yb = tf.math.minimum(height, y + sq_height // 2)
        xa = tf.math.maximum(0, x - sq_width // 2)
        xb = tf.math.minimum(width, x + sq_width // 2)
        # DROPOUT IMAGE
        one = image[ya:yb, 0:xa, :]
        two = tf.zeros([yb - ya, xb - xa, 3])
        three = image[ya:yb, xb:width, :]
        middle = tf.concat([one, two, three], axis=1)
        image = tf.concat([image[0:ya, :, :], middle, image[yb:height, :, :]], axis=0)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    image = tf.reshape(image, [height, width, 3])
    return image


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
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0):
        return image  # no action

    h, w, c = image.shape

    noise = tf.random.normal(shape=image.shape, mean=tf.reduce_mean(image), stddev=tf.math.reduce_std(image) * std)
    image = image + noise
    image = tf.reshape(image, [h, w, c])
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
