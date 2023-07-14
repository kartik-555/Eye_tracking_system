import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_dir = 'eye_dataset/Eye dataset'
image_height = 64
image_width = 64
num_channels = 3

labels = os.listdir(data_dir)

images = []
labels_list = []

for label in labels:
    label_dir = os.path.join(data_dir, label)

    # Get the list of image file names in the label folder
    image_files = os.listdir(label_dir)

    # Iterate through each image file
    for image_file in image_files:
        image_path = os.path.join(label_dir, image_file)

        # Read and resize the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_height, image_width))

        # Normalize the image
        image = image.astype('float32') / 255.0

        # Append the image and corresponding label to the lists
        images.append(image)
        labels_list.append(label)

images = np.array(images)
labels = np.array(labels_list)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()

train_images = np.array(train_images)
train_labels = np.array(train_labels)

train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

train_labels = to_categorical(train_labels, num_classes=4)
val_labels = to_categorical(val_labels, num_classes=4)
test_labels = to_categorical(test_labels, num_classes=4)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50, batch_size=64, validation_data=(val_images, val_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

model.save('eye_model.h5')
