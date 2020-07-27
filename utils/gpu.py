import tensorflow as tf
import subprocess as sp
import os
import warnings
import logging
from .logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)


class MemoryCheck(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mem = sp.check_output('nvidia-smi | grep python', shell=True).split()[-2].decode('utf-8')
        print(' ' + mem)


def tensorflow_defult_setting():
    import os
    import sys
    import warnings
    import logging

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    tf.get_logger().setLevel(logging.ERROR)

    sys.stdout = Logger(filename="log")

    print("TensorFlow Version: ", tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("GPU Information: ", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# import tensorflow as tf
# import subprocess as sp
# memory_usage = []
#
#
# class MemoryCheck(tf.keras.callbacks.Callback):
#     def on_batch_end(self, batch, logs=None):
#         mem = sp.check_output('nvidia-smi | grep python', shell=True).split()[-2].decode('utf-8')
#         memory_usage.append(int(mem[:-3]))
#         print(' ' + mem)
#
#
# mem_check = MemoryCheck()
#
# import matplotlib.pyplot as plt
#
# plt.plot(memory_usage)

if __name__ == '__main__':
    tensorflow_defult_setting()
