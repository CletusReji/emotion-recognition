import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set dataset directory paths
train_dir = "archive/train"
test_dir = "archive/test"

# Define image size and batch size
img_size = (48, 48)
batch_size = 64

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for testing

# Load images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

# Define model architecture
model = Sequential([
    # First Convolutional Block
    Conv2D(64, (3,3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),
    
    # Second Convolutional Block
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    # Third Convolutional Block
    Conv2D(256, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    # Fully Connected Layers
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # 7 classes (emotions)
])

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Model Summary
model.summary()

# Train the model
epochs = 30

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs
)

# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')

plt.show()

model.save("facial_emotion_model.keras")
