import tensorflow as tf


def build_tpu_model(create_model, use_tpu=True):
    if use_tpu:
        # Create distribution strategy
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)

        # Create model
        with strategy.scope():
            model = create_model()
    else:
        model = create_model()

    return model
