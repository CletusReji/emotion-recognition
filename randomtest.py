#import os
import tensorflow as tf

# Set the environment variable to disable oneDNN custom operations
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print(tf.__version__)
print(tf.keras.__version__)