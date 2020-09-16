import tensorflow as tf


def _log(s, verbose):
    if verbose:
        print(s)


def get_device_strategy(device, verbose=True):
    """
    Returns the distributed strategy object, the tune policy anb the number of replicas.

    Inputs:
        device: string indicating the device to run on. Possible values are "TPU", "GPU", "CPU"
        verbose: Whether to print the messages or not
    Outputs:
        strategy: The distributed strategy object
        tune: the auto tune object
        replicas: int indicating the number of replicas, to adjust batch size and learning rate
    """
    device = device.upper()
    v = tf.__version__

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
                _log("initializing  TPU ...", verbose)
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.TPUStrategy(tpu) if v >= '2.3.0' else tf.distribute.experimental.TPUStrategy(
                    tpu)
                _log("TPU initialized", verbose)
            except _:
                _log("failed to initialize TPU", verbose)
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

    Inputs:
        Tpu: the TPU cluster
    """
    if tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
