import tensorflow as tf


def _log(s, verbose):
    """
    Prints a message on the screen
    """
    if verbose:
        print(s)


def get_device_strategy(device, half=False, XLA=False, verbose=True):
    """
    Returns the distributed strategy object, the tune policy anb the number of replicas.

    Parameters
    ----------
        device : str
            Possible values are "TPU", "GPU", "CPU"
        verbose : bool
            Whether to print the output messages or not
    Returns
    -------
    tf.distribute.TPUStrategy
        The distributed strategy object
    int
        The auto tune constant
    int
        Number of TPU cores, to adjust batch size and learning rate
    tf.distribute.cluster_resolver.TPUClusterResolver
        The tpu object
    """
    device = device.upper()
    v = tf.__version__
    tpu = None

    if device == "TPU":
        _log("connecting to TPU...", verbose)
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            _log('Running on TPU ' + tpu.master(), verbose)
        except ValueError:
            _log("Could not connect to TPU", verbose)
            tpu = None

        if tpu:
            try:
                _log("initializing TPU ...", verbose)
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.TPUStrategy(tpu) if v >= '2.3.0' else tf.distribute.experimental.TPUStrategy(
                    tpu)
                _log("TPU initialized", verbose)

                if half:
                    from tensorflow.keras.mixed_precision import experimental as mixed_precision
                    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
                    mixed_precision.set_policy(policy)
                    print('Mixed precision enabled')
                if XLA:
                    tf.config.optimizer.set_jit(True)
                    print('Accelerated Linear Algebra enabled')

            except:
                _log("failed to initialize TPU", verbose)
                device = "GPU"
        else:
            device = "GPU"

    if device != "TPU":
        _log("Using default strategy for CPU and single GPU", verbose)
        strategy = tf.distribute.get_strategy()

    if device == "GPU":
        _log("Num GPUs Available: " + str(len(tf.config.experimental.list_physical_devices('GPU') if v < '2.1.0' else
                                              tf.config.list_physical_devices('GPU'))), verbose)

    tune = tf.data.experimental.AUTOTUNE
    replicas = strategy.num_replicas_in_sync
    _log(f'REPLICAS: {replicas}', verbose)
    return strategy, tune, replicas, tpu


def init_tpu(tpu):
    """
    Re-initializes the TPU cluster, useful to clean up memory

    Parameters
    ----------
    tf.distribute.cluster_resolver.TPUClusterResolver
        The TPU cluster
    """
    if tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
