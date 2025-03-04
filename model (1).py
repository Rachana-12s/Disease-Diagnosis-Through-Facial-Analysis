import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight

# Function to load and preprocess dataset with data augmentation
def load_data(base_dir, img_size=(150, 150)):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,  # 20% for validation
        rotation_range=20,      # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2, # Randomly shift images vertically
        shear_range=0.2,        # Shear transformation
        zoom_range=0.2,         # Random zoom
        horizontal_flip=True,   # Randomly flip images
        fill_mode='nearest'     # Fill in missing pixels
    )
    
    # Training data generator
    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation data generator
    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

# Define CNN model architecture
def build_model(img_size=(150, 150, 3), num_classes=3):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=img_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense layers for classification
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # To avoid overfitting

    model.add(Dense(num_classes, activation='softmax'))  # Output layer for three classes

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Path to the dataset directory
base_dir = "C:/Users/coelh/Downloads/disease_Diagnosis_face/dataset1"

# Load data
train_generator, validation_generator = load_data(base_dir)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Build the model
model = build_model(num_classes=3)  # Build the model with 3 output classes

# Train the model and store the history
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    class_weight=class_weights,  # Use class weights
    verbose=1  # Ensure you see the output
)

# Save the trained model
model.save('disease_detection_model.keras')

# Plotting training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
