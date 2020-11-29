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
        4x4 transformation matrix for 3D transformations
    """

    # CONVERT DEGREES TO RADIANS
    # rotation = _check_rotation_arg(rotation)

    def get_4x4_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [4, 4])

    # ROTATION MATRIX
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    # X axis
    r = math.pi * rotation[0:1] / 180.
    cx = tf.math.cos(r)
    sx = tf.math.sin(r)
    rx = get_4x4_mat([one, zero, zero, zero,
                      zero, cx, -sx, zero,
                      zero, sx, cx, zero,
                      zero, zero, zero, one])

    # Y axis
    r = math.pi * rotation[1:2] / 180.
    cy = tf.math.cos(r)
    sy = tf.math.sin(r)
    ry = get_4x4_mat([cy, zero, sy, zero,
                      zero, one, zero, zero,
                      -sy, zero, cy, zero,
                      zero, zero, zero, one])

    # Z axis
    r = math.pi * rotation[2:] / 180.
    cz = tf.math.cos(r)
    sz = tf.math.sin(r)
    rz = get_4x4_mat([cz, -sz, zero, zero,
                      sz, cz, zero, zero,
                      zero, zero, one, zero,
                      zero, zero, zero, one])

    rand = tf.random.uniform([], minval=0, maxval=3, dtype=tf.int32)
    if rand == 0:
        rotation_matrix = rx
    else:
        if rand == 1:
            rotation_matrix = ry
        else:
            rotation_matrix = rz

    # SHEAR MATRIX
    shear = math.pi * shear / 180.
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    # shear_matrix = get_4x4_mat([one, s2, zero,
    #                            zero, c2, zero,
    #                            zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_4x4_mat([one / x_zoom, zero, zero, zero,
                               zero, one / y_zoom, zero, zero,
                               zero, zero, one / z_zoom, zero,
                               zero, zero, zero, one])
    # SHIFT MATRIX
    shift_matrix = get_4x4_mat([one, zero, zero, x_shift,
                                zero, one, zero, y_shift,
                                zero, zero, one, z_shift,
                                zero, zero, zero, one])

    return K.dot(rotation_matrix,
                 K.dot(zoom_matrix, shift_matrix))


def transform3d(obj, dimension, rotation=None, shear=2.0, x_zoom=8.0, y_zoom=8.0, z_zoom=8.0, x_shift=8.0, y_shift=8.0,
                z_shift=8.0, prob=1.0):
    """
    Rotates, shears, zooms and shift an single object, not a batch of them.

    Parameters
    ----------
    image : tf.Tensor of shape [h,w,d,c]
        A single image to be transformed
    dimension : int
        Dimension in pixels of the squared image
    rotation : float or list of floats
        Degrees to rotate
    shear : float
        Degrees to shear
    x_zoom : float
        height zoom ratio
    y_zoom : float
        width zoom ratio
    z_zoom : float
        width zoom ratio
    x_shift : float
        height shift ratio
    y_shift : float
        width shift ratio
    z_shift : float
        width shift ratio
    prob : float
        probabilities to apply transformations

    Returns
    -------
    tf.Tensor
        A transformed object
    """

    XDIM = dimension % 2

    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if P == 0:
        return tf.reshape(obj, [dimension, dimension, dimension, 3])  # no action

    rotation = _check_rotation_arg(rotation)


    rot = rotation * tf.random.normal([3], dtype='float32')
    shr = shear * tf.random.normal([1], dtype='float32')
    x_zoom = 1.0 + tf.random.normal([1], dtype='float32') / x_zoom
    y_zoom = 1.0 + tf.random.normal([1], dtype='float32') / y_zoom
    z_zoom = 1.0 + tf.random.normal([1], dtype='float32') / z_zoom
    x_shift = x_shift * tf.random.normal([1], dtype='float32')
    y_shift = y_shift * tf.random.normal([1], dtype='float32')
    z_shift = z_shift * tf.random.normal([1], dtype='float32')

    # print(rot,shr,x_zoom,y_zoom,z_zoom)

    # GET TRANSFORMATION MATRIX
    m = get_mat3d(rot, shr, x_zoom, y_zoom, z_zoom, x_shift, y_shift, z_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(dimension // 2, -dimension // 2, -1), dimension * dimension)
    y = tf.tile(tf.repeat(tf.range(dimension // 2, -dimension // 2, -1), dimension), [dimension])
    z = tf.tile(tf.range(-dimension // 2, dimension // 2), [dimension * dimension])
    c = tf.ones([dimension * dimension * dimension], dtype='int32')
    idx = tf.stack([x, y, z, c])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -dimension // 2 + XDIM + 1, dimension // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([dimension // 2 - idx2[0,], dimension // 2 - idx2[1,], dimension // 2 - 1 + idx2[2,]])
    d = tf.gather_nd(obj, tf.transpose(idx3))

    return tf.reshape(d, [dimension, dimension, dimension, 3])


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
    tf.Tensor
        3x3 transformation matrix for 2D transformations
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


def transform2d(image, dimension, rotation=180.0, shear=2.0, hzoom=8.0, wzoom=8.0, hshift=8.0, wshift=8.0, prob=0.5):
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
    hzoom : float
        height zoom ratio
    wzoom : float
        width zoom ratio
    hshift : float
        height shift ratio
    wshift : float
        width shift ratio
    prob : float
        probabilities to apply transformations

    Returns
    -------
    tf.Tensor
        A transformed image
    """

    XDIM = dimension % 2

    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if P == 0:
        return tf.reshape(image, [dimension, dimension, 3])  # no action

    rot = rotation * tf.random.normal([1], dtype='float32')
    shr = shear * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom
    h_shift = hshift * tf.random.normal([1], dtype='float32')
    w_shift = wshift * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat2d(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(dimension // 2, -dimension // 2, -1), dimension)
    y = tf.tile(tf.range(-dimension // 2, dimension // 2), [dimension])
    z = tf.ones([dimension * dimension], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -dimension // 2 + XDIM + 1, dimension // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([dimension // 2 - idx2[0,], dimension // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [dimension, dimension, 3])


def _reconstruct2D(a, b, xa, xb, ya, yb):
    one = a[ya:yb, :xa, :]
    two = b[ya:yb, xa:xb, :]
    three = a[ya:yb, xb:, :]
    middle = tf.concat([one, two, three], axis=1)
    return tf.concat([a[:ya, :, :], middle, a[yb:, :, :]], axis=0)


def _reconstruct3D(a, b, xa, xb, ya, yb, za, zb):
    one = a[ya:yb, :xa, :, :]
    two_a = a[ya:yb, xa:xb, :za, :]
    two = b[ya:yb, xa:xb, za:zb, :]
    two_b = a[ya:yb, xa:xb, zb:, :]
    three = a[ya:yb, xb:, :, :]
    two = tf.concat([two_a, two, two_b], axis=2)
    middle = tf.concat([one, two, three], axis=1)
    return tf.concat([a[:ya, :, :, :], middle, a[yb:, :, :, :]], axis=0)


def _points(dim, location, size):
    a = tf.math.maximum(0, location - size // 2)
    b = tf.math.minimum(dim, location + size // 2)
    return a, b


def dropout(image, prob=0.75, ct=8, sz=0.2, rank=2):
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
    rank : int
        values must be 2 (image) or 3 (3d shape)

    Returns
    -------
        image with ct squares of side size sz*dimension removed
    """

    if (rank != 2) & (rank != 3):
        raise Exception('Rank must be 2 or 3')

    if rank == 2:
        h, w, c = image.shape
    else:
        h, w, d, c = image.shape

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast(tf.random.uniform([], 0, 1) < prob, tf.int32)
    if (P == 0) | (ct == 0) | (sz == 0):
        if rank == 2:
            image = tf.reshape(image, [h, w, 3])
        else:
            image = tf.reshape(image, [h, w, d, 3])
        return image  # no action

    # Extract dimension
    if rank == 2:
        h, w, c = image.shape
    else:
        h, w, d, c = image.shape

    # Calculate square/box size
    sq_height = tf.cast(sz * h, tf.int32) * P
    sq_width = tf.cast(sz * w, tf.int32) * P

    if rank == 3:
        sq_depth = tf.cast(sz * d, tf.int32) * P

    # generate random black squares
    for k in range(ct):
        # Choose random location
        x = tf.cast(tf.random.uniform([], 0, w), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, h), tf.int32)

        # Compute square / cube
        ya, yb = _points(h, y, sq_height)
        xa, xb = _points(w, x, sq_width)

        # Include third dimension for 3D
        if rank == 3:
            z = tf.cast(tf.random.uniform([], 0, d), tf.int32)
            za, zb = _points(h, z, sq_depth)

        # Dropout Image
        if rank == 2:
            image = _reconstruct2D(image, tf.zeros_like(image), xa, xb, ya, yb)
        else:
            image = _reconstruct3D(image, tf.zeros_like(image), xa, xb, ya, yb, za, zb)

    # Reshape hack so TPU compiler knows shape of output tensor
    if rank == 2:
        image = tf.reshape(image, [h, w, 3])
    else:
        image = tf.reshape(image, [h, w, d, 3])

    return image


def _mixup_labels(shape, label1, label2, n_classes, a):
    if len(shape) == 1:
        lab1 = tf.one_hot(label1, n_classes)
        lab2 = tf.one_hot(label2, n_classes)
    else:
        lab1 = tf.cast(label1, dtype=tf.float32)
        lab2 = tf.cast(label2, dtype=tf.float32)
    return (1 - a) * lab1 + a * lab2


def cutmix(batch, label, batch_size=32, prob=1.0, dimension=256, n_classes=1, n_labels=None):
    """
    Cutmix randomly remove squares from training images

    Parameters
    ----------
    batch : tf.Tensor
        batch of [b,dim,dim,3] or [b,dim,dim,dim,3]
    label : tf.tensor
        batch of shape [b,] if labels are integer, or [b,n_classes] if format is one-hot
    prob : float
        probability to perform dropout
    batch_size : int
        batch size
    dimension : int
        dimension of the data
    n_classes : int
        number of classes
    rank : int
        values must be 2 (image) or 3 (3d shape)

    Returns
    -------
    tf.Tensor
        A batch of images with Cutmix applied
    """

    rank = len(batch.shape) - 2

    imgs = []
    labs = []
    for j in range(batch_size):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= prob, tf.int32)

        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
        WIDTH = tf.cast(dimension * tf.math.sqrt(1 - b), tf.int32) * P

        # Choose random location
        x = tf.cast(tf.random.uniform([], 0, dimension), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, dimension), tf.int32)

        # Compute square / cube
        ya, yb = _points(dimension, y, WIDTH)
        xa, xb = _points(dimension, x, WIDTH)

        # Include third dimension for 3D
        if rank == 3:
            z = tf.cast(tf.random.uniform([], 0, dimension), tf.int32)
            za, zb = _points(dimension, z, WIDTH)

        # Choose Random Image to Cutmix with
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)

        # Make Cutmix Image
        if rank == 2:
            image = _reconstruct2D(batch[j], batch[k], xa, xb, ya, yb)
        elif rank == 3:
            image = _reconstruct3D(batch[j], batch[k], xa, xb, ya, yb, za, zb)
        else:
            raise Exception(f"Rank incorrect. Should be 2 or 3, but it is {rank}") 
        imgs.append(image)

        # Make Cutmix Label
        a = tf.cast((WIDTH ** rank) / (dimension ** rank), tf.float32)
        labs.append(_mixup_labels(label.shape, label[j], label[k], n_classes, a))

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    if rank == 2:
        image2 = tf.reshape(tf.stack(imgs), (batch_size, dimension, dimension, 3))
    elif rank == 3:
        image2 = tf.reshape(tf.stack(imgs), (batch_size, dimension, dimension, dimension, 3))
    else:
        raise Exception(f"Rank incorrect. Should be 2 or 3, but it is {rank}") 

    if n_labels:
        label2 = tf.reshape(tf.stack(labs), (batch_size, n_labels))
    else:
        label2 = tf.reshape(tf.stack(labs), (batch_size, n_classes))
    return image2, label2


def mixup(batch, label, batch_size=32, prob=1.0, dimension=256, n_classes=1, n_labels=None):
    """
    Mixup randomly mixes data from two samples

    Parameters
    ----------
    batch : tf.Tensor
        batch of [b,dim,dim,3] or [b,dim,dim,dim,3]
    label : tf.tensor
        batch of shape [b,] if labels are integer, or [b,n_classes] if format is one-hot
    prob : float
        probability to perform dropout
    batch_size : int
        batch size
    dimension : int
        dimension of the data
    n_classes : int
        number of classes
    rank : int
        values must be 2 (image) or 3 (3d shape)

    Returns
    -------
    tf.Tensor
        A batch of images with Mixup applied
    """

    rank = len(batch.shape) - 2
    #batch_size = batch.shape[0]

    imgs = []
    labs = []
    for j in range(batch_size):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= prob, tf.float32)
        # Choose Random
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
        # Make mixup image
        img1 = batch[j,]
        img2 = batch[k,]
        imgs.append((1 - a) * img1 + a * img2)

        # Make Cutmix Label
        labs.append(_mixup_labels(label.shape, label[j], label[k], n_classes, a))

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (batch_size, dimension, dimension, 3)) if rank == 2 else tf.reshape(tf.stack(imgs), (
    batch_size, dimension, dimension, dimension, 3))

    if n_labels:
        label2 = tf.reshape(tf.stack(labs), (batch_size, n_labels))
    else:
        label2 = tf.reshape(tf.stack(labs), (batch_size, n_classes))
    return image2, label2


def spec_augmentation(image, prob=0.66, time_drop_width=0.0625, time_stripes_num=2, freq_drop_width=0.125,
                      freq_stripes_num=2, height=64, width=501):
    """
    Add white noise to object or image

    Parameters
    ----------
    image : tf.Tensor
        image of size [height,width,3] not a batch of [b,dim,dim,3]
    prob : float
        probability to perform dropout
    time_drop_width : float

    time_stripes_num : int

    freq_drop_width : float

    freq_stripes_num : int

    height : int

    width : int


    Returns
    -------
    tf.Tensor
        input image or object with added white noise
    """
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
    """
    Add white noise to an horizontal band in an image

    Parameters
    ----------
    image : tf.Tensor
        image of size [height,width,3] not a batch of [b,dim,dim,3]
    prob : float
        probability to perform dropout
    std : size
        Number of standard deviations to calculate noise with
    band_height : float
        Percentage of the height of the band

    Returns
    -------
    tf.Tensor
        input image or object with added noise
    """
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