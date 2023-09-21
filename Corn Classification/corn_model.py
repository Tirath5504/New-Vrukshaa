import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directories for train, validation, and test sets
train_directory = "C:/Users/TIRATH/Documents/New_Vrukshaa/Corn Classification/train_test_validate/train"
valid_directory = "C:/Users/TIRATH/Documents/New_Vrukshaa/Corn Classification/train_test_validate/validate"
test_directory = "C:/Users/TIRATH/Documents/New_Vrukshaa/Corn Classification/train_test_validate/test"

# Define image dimensions and batch size
image_height, image_width = 224, 224
batch_size = 32

# Create data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,    # Augmentation: random rotation
    width_shift_range=0.2,  # Augmentation: random horizontal shift
    height_shift_range=0.2,  # Augmentation: random vertical shift
    horizontal_flip=True,  # Augmentation: random horizontal flip
    shear_range=0.2,        # Augmentation: random shear
    zoom_range=0.2,         # Augmentation: random zoom
    fill_mode="nearest"    # Augmentation: fill mode for pixels outside boundaries
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only rescale for validation and testing

train_data = train_datagen.flow_from_directory(
    train_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# Build a simple CNN model
model = Sequential[
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax') ]

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10  # Adjust the number of epochs
history = model.fit(train_data, epochs=epochs, validation_data=valid_data)

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_data = test_datagen.flow_from_directory(
    test_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# After training
model.save("corn_model.keras")

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy}")

validation_loss, validation_accuracy = model.evaluate(valid_data)
print(f"Validation Loss: {validation_loss}")
print(f"Validation Accuracy: {validation_accuracy}")
