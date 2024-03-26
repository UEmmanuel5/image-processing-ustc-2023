# %%
import pandas as pd
import os
import glob as gb
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import numpy as np
import time

# Constants
TRAIN_DIR = "class/train"
TEST_DIR = "class/test"
BATCH_SIZE = 2048


# %%
# Function to view a random image from a target class
def view_random_image(target_dir, target_class):
    target_folder = os.path.join(target_dir, target_class)
    random_image = random.choice(os.listdir(target_folder))
    img_path = os.path.join(target_folder, random_image)
    img = mpimg.imread(img_path)
    plt.imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
    plt.title(target_class)
    plt.axis('off')
    print(f"Image shape: {img.shape}")
    return img

# Display random images from different classes
class_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
plt.figure(figsize=(20, 10))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    class_name = random.choice(class_names)
    img = view_random_image(target_dir="class/train/", target_class=class_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')

plt.show()


# %%
# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# %%
# CNN Model
classifier = Sequential()
classifier.add(Conv2D(16, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Additional Convolutional Layer 1 (layer 3)
classifier.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Additional Convolutional Layer 2 (layer 4)
classifier.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Additional Convolutional Layer 2 (layer 5)
# classifier.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Additional Convolutional Layer 2 (layer 6)
# classifier.add(Conv2D(512, (3, 3), activation='relu', strides=(1, 1), padding='same'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=7, activation='softmax'))  

# # Compile the CNN
# optimizer = Adam()
# classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
lr= 0.001
optimizer = Adam(learning_rate= lr )  # Using Adam optimizer
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# Model Summary
classifier.summary()

# %%
# Model Training
history = classifier.fit(
    training_set,
    epochs=100,
    validation_data=test_set
)


# %%
# # Save the model with a timestamp
# timestamp = time.strftime("%Y%m%d%H%M%S")
exp = 2048
model_filename = f'model_{exp}.h5'
classifier.save(model_filename)
print(f'Model saved as {model_filename}')

# Plot training history
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# %%
# Classifying UTK Images
import os
import shutil

# Function to predict and organize images
def organize_images(input_folder, output_folder, model, class_names):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Construct the full path for the image
                image_path = os.path.join(root, file)

                # Perform prediction on the image
                image = cv2.imread(image_path)
                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((128, 128))
                expand_input = np.expand_dims(resize_image, axis=0)
                input_data = np.array(expand_input) / 255.0
                pred = model.predict(input_data)
                predicted_class = np.argmax(pred)
                predicted_label = class_names[predicted_class]

                # Create the output folder if it doesn't exist
                output_class_folder = os.path.join(output_folder, predicted_label)
                os.makedirs(output_class_folder, exist_ok=True)

                # Copy the image to the appropriate class folder
                shutil.copy(image_path, os.path.join(output_class_folder, file))

# Example usage
input_folder_path = "class/UTKFace"
output_folder_path = "class/labels2048"

organize_images(input_folder_path, output_folder_path, classifier, class_names)


# # %%
# # Prediction on a new image
# new_image_path = "class/UTKFace/surprise/adult.jpg"
# image = cv2.imread(new_image_path)
# image_fromarray = Image.fromarray(image, 'RGB')
# resize_image = image_fromarray.resize((128, 128))
# expand_input = np.expand_dims(resize_image, axis=0)
# input_data = np.array(expand_input) / 255.0

# pred = classifier.predict(input_data)
# result = np.argmax(pred)
# print(f'Predicted class index: {result}')
# print(f'Predicted class: {class_names[result]}')



