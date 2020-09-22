import tensorflow as tf

def bam_module(input, reduction_ratio=8, dilation_rate=4):
    """
    BAM: Bottleneck Attention Module
    https://arxiv.org/pdf/1807.06514.pdf
    """
    n, h, w, c = input.shape  # (N,H,W,C)

    # Channel attention module
    x1 = tf.keras.layers.GlobalAveragePooling2D()(input)
    x1 = tf.keras.layers.Dense(units=c/reduction_ratio, activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dense(units=c, activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Reshape(target_shape=[1,1,-1])(x1)

    # Spatial Attention module
    x2 = tf.keras.layers.Conv2D(filters=c/reduction_ratio, kernel_size=1, padding='same')(input)
    x2 = tf.keras.layers.Conv2D(filters=c/reduction_ratio, kernel_size=3, dilation_rate=dilation_rate, padding='same')(x2)
    x2 = tf.keras.layers.Conv2D(filters=c/reduction_ratio, kernel_size=3, dilation_rate=dilation_rate, padding='same')(x2)
    x2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)

    # Combine attention modules
    x = tf.keras.layers.Add()([x1, x2])
    x = tf.keras.layers.Activation('sigmoid')(x)

    # Applies attention to input
    x = input * x
    x = input + x

    return x

def cbam_module(input, reduction_ratio=8,):
    """
    CBAM Convolutional Block Attention Module
    https://arxiv.org/pdf/1807.06521.pdf

    """
    n, h, w, c = input.shape  # (N,H,W,C)

    dense = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(c / reduction_ratio),
                                 tf.keras.layers.Activation('relu'),
                                 tf.keras.layers.Dense(c)])

    # Channel attention module
    x1 = tf.keras.layers.GlobalAveragePooling2D()(input)
    x1 = dense(x1)

    x2 = tf.keras.layers.GlobalMaxPooling2D()(input)
    x2 = dense(x2)

    x = tf.keras.layers.Add()([x1, x2])
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Reshape(target_shape=[1,1,-1])(x)

    x = x * input

    # Spatial Attention Map
    xavg = tf.expand_dims(tf.math.reduce_mean(input_tensor=x, axis=3), axis=3)
    xmax = tf.expand_dims(tf.math.reduce_max(input_tensor=x, axis=3), axis=3)

    x = tf.keras.layers.concatenate([xavg, xmax])
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    x = x * input
    return x







