import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=0.1)

train_data = data_gen.flow_from_directory('D:/Dow', target_size=(48, 48), color_mode='grayscale', class_mode='categorical')
