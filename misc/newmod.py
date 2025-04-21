import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set dataset directory paths
train_dir = "archive/train"
test_dir = "archive/test"

# Define image size and batch size
img_size = (48, 48)
batch_size = 64

# Data Augmentation for Training & Validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.15,  # 15% validation split
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    channel_shift_range=0.1,
    fill_mode="nearest"
)

# Training Data (85%)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Validation Data (15%)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Test Data (Separate Dataset, No Augmentation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False  # Important for confusion matrix
)

# Define CNN Model Architecture
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3,3), activation="relu", input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Second Convolutional Block
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Third Convolutional Block
    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Fourth Convolutional Block
    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    # Fully Connected Layers
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.4),
    Dense(7, activation="softmax")  # 7 classes for emotion detection
])

# Compile Model with Lower Learning Rate
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0005),  # Reduced LR for better convergence
    metrics=["accuracy"]
)

# Model Summary
model.summary()

# Callbacks for Better Training
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max")
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001)

# Train the Model
epochs = 50

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Plot Accuracy & Loss Curves
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

# Save Model
model.save("newerfacemodel.keras")

# Evaluate the Model on the Test Set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n🔹 Test Accuracy: {test_accuracy:.4f}")
print(f"🔹 Test Loss: {test_loss:.4f}")

# Predict on Test Data
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=test_generator.class_indices.keys(), 
            yticklabels=test_generator.class_indices.keys())

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()