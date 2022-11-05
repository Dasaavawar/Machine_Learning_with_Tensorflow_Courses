#Neural Networks with TensorFlow

#Creating a Neural Network

#Imports and Setup
#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Dataset
#For this tutorial we will use the MNIST Fashion Dataset. This is a dataset that is included in keras.
#This dataset includes 60,000 images for training and 10,000 images for validation/testing.
fashion_mnist = keras.datasets.fashion_mnist  #load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#split into tetsing and training

#Let's have a look at this data to see what we are working with.
train_images.shape

train_images[0,23,23]  #let's have a look at one pixel

#This means we have a grayscale image as there are no color channels.
train_labels[:10]  # let's have a look at the first 10 training labels

#We'll create an array of label names to indicate which is which.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Finally let's look at what some of these images look like!
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

#Data Preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

#Building the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

#Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the Model
model.fit(train_images, train_labels, epochs=10) 
#we pass the data, labels and epochs and watch the magic!

#Evaluating the Model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

print('Test accuracy:', test_acc)

#Making Predictions
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]

#Verifying Predictions
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)