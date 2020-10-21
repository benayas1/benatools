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

def cbam_module(input, reduction_ratio=8, kernel_size=7):
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
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    x = x * input
    return x

def cbam_module_3d(input, reduction_ratio=8, kernel_size=7):
    """
    CBAM Convolutional Block Attention Module
    https://arxiv.org/pdf/1807.06521.pdf

    """
    n, h, w, d, c = input.shape  # (N,H,W,D,C)

    dense = tf.keras.Sequential([tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(c / reduction_ratio),
                                 tf.keras.layers.Activation('relu'),
                                 tf.keras.layers.Dense(c)])

    # Channel attention module
    x1 = tf.keras.layers.GlobalAveragePooling3D()(input)
    x1 = dense(x1)

    x2 = tf.keras.layers.GlobalMaxPooling3D()(input)
    x2 = dense(x2)

    x = tf.keras.layers.Add()([x1, x2])
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Reshape(target_shape=[1,1,1,-1])(x)

    x = x * input

    # Spatial Attention Map
    xavg = tf.expand_dims(tf.math.reduce_mean(input_tensor=x, axis=3), axis=3)
    xmax = tf.expand_dims(tf.math.reduce_max(input_tensor=x, axis=3), axis=3)

    x = tf.keras.layers.concatenate([xavg, xmax])
    x = tf.keras.layers.Conv3D(filters=1, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    x = x * input
    return x


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    scores = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = scores / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # d_model must be divisible by num_heads
        assert d_model % self.num_heads == 0

        # depth is the depth of each head
        self.depth = d_model // self.num_heads

        # three dense layers to generate query, keys and values
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # final dense layer
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        # calculate batch size
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        # calculate q, v and k
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # split each time step into num_heads heads
        # seq_len_q is the same as seq_len_k
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Calculate attention
        # scaled attention shape has 2 dimensions inverted
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # Transpose dimension 1 and 2 to have seq_len back in dimension 1
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # Now we have to concat back all the different heads, thus removing one dimension
        concat_attention = tf.reshape(scaled_attention, (batch_size, seq_len, self.d_model))  # (batch_size, seq_len_q, d_model)

        # Finally apply another dense layer to the concatenated output
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def get_config(self):
        config = {
            'num_heads': self.num_heads,
            'd_model': self.d_model,
        }
        return config





