# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import random

#print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

def hmix_deprecated(image, index):
  for i in range(len(image)):
    image[i] = np.concatenate((image[i][::-1][0:len(image[i])-index], image[i][0:index]))

def vmix_noteither(image, index):
  top = image[0:index]
  bottom = image[index:len(image)]
  if top == []:
    image[:] = bottom[::-1]
  else:
    image[:] = np.vstack((bottom[::-1], top))

def hmix(image, index, index2):
  original = np.copy(image)
  width = len(image[0])
  height = len(image)
  for i in range(width):
    j = (i + index2) % height
    image[i][index:width] = original[j][index:width]


def vmix(image, index, index2):
  original = np.copy(image)
  width = len(image[0])
  height = len(image)
  for i in range(index,height):
    image[i][0:width-index2] = original[i][index2:width]
    image[i][width-index2:width] = original[i][0:index2]


image = np.copy(train_images[0])
width = len(image[0])
height = len(image)
for i in range(30):
  hmix(image, random.randint(0,width), random.randint(0,height))
  vmix(image, random.randint(0,width), random.randint(0,height))

plt.figure()
plt.imshow(image)
plt.colorbar()
plt.grid(False)
plt.show()

