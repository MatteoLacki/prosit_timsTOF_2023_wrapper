import functools


@functools.cache
def load_model(path):
    import tensorflow as tf  # on purspose here, do not dare to move it on top of the module!

    return tf.saved_model.load(path)
