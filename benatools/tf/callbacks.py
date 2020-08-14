import tensorflow as tf


def get_lr_callback(batch_size=8, lr_start=0.000005, lr_max=0.00000125, lr_min=0.000001, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8):
    """ Train schedule for transfer learning. The learning rate starts near zero, then increases to a maximum, then decays over time.
        A good practice to follow is to increase maximum learning rate as batch size increase

        Input:
            batch_size
            lr_start: initial learning rate value
            lr_max: maximum learning rate.
            lr_min: minimum learning rate.
            lr_ramp_ep: number of epochs of ramp up
            lr_sus_ep: number of epochs of plateau
            lr_decay: decay [0,1]
    """

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback