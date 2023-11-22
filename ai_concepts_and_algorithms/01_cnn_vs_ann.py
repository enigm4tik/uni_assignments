# Machine Learning 2: AI Concepts and Algorithms
# Assignment #01: Convolutional Neural Networks vs. Fully connected Artificial Neural Networks

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# CIFAR10 dataset containing of 32x32 images in 10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

## Convolutional Model
# Consisting of convolutional layers and pooling layers

cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dense(10))
cnn_model.summary()

cnn_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

cnn_history = cnn_model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

plt.figure(0)
plt.title('Convolutional Neural Network')
plt.plot(cnn_history.history['accuracy'], label='Training Set')
plt.plot(cnn_history.history['val_accuracy'], label='Test Set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
## Fully connected Artificial Neural Network
# using dense layers (connected to all neurons in the previous layer)

ann_model = models.Sequential()
ann_model.add(layers.Dense(80, input_shape=(32, 32, 3), activation='relu'))
ann_model.add(layers.Flatten())
ann_model.add(layers.Dense(64, activation='relu'))
ann_model.add(layers.Dense(32, activation='relu'))
ann_model.add(layers.Dense(10))

ann_model.build()
ann_model.summary()

ann_model.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

ann_history = ann_model.fit(train_images, train_labels, epochs=10,
                     validation_data=(test_images, test_labels))

plt.figure(1)
plt.title("Artificial Neural Network")
plt.plot(ann_history.history['accuracy'], label='Training Set')
plt.plot(ann_history.history['val_accuracy'], label='Test Set')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()
