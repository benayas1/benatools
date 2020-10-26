import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

def get_dataset(files, cfg, augment = False, shuffle = False, repeat = False,
                labeled=True, batch_size=8, read_function = None):

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(lambda example: read_function(example), num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_function(example), num_parallel_calls=AUTO)

    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, cfg=cfg),
                                               imgname_or_label),
                num_parallel_calls=AUTO)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return ds


def prepare_image(img, cfg=None, augment=True):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg['read_size'], cfg['read_size']])
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = transform(img, cfg)
        img = tf.image.random_crop(img, [cfg['crop_size'], cfg['crop_size'], 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    else:
        img = tf.image.central_crop(img, cfg['crop_size'] / cfg['read_size'])

    img = tf.image.resize(img, [cfg['net_size'], cfg['net_size']])
    img = tf.reshape(img, [cfg['net_size'], cfg['net_size'], 3])
    return img





