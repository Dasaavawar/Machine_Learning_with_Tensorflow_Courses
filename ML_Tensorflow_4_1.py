#Deep Computer Vision

#Imports and Setup
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Convolutional Neural Network

#Dataset
#The problem we will consider here is classifying 10 different everyday objects.
#The dataset we will use is built into tensorflow and called the CIFAR Image Dataset.
#It contains 60,000 32x32 color images with 6000 images of each class.

#The labels in this dataset are the following:
#-Airplane
#-Automobile
#-Bird
#-Cat
#-Deer
#-Dog
#-Frog
#-Horse
#-Ship
#-Truck

#LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#Let's look at a one image
IMG_INDEX = 7  #change this to look at other images

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

#CNN Architecture

#A common architecture for a CNN is a stack of Conv2D and MaxPooling2D layers followed by a few denesly connected layers.
#To idea is that the stack of convolutional and maxPooling layers extract the features from the image.
#Then these features are flattened and fed to densly connected layers that determine the class of an image based on the presence of features.

#We will start by building the Convolutional Base.
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Layers
#Let's have a look at our model so far
model.summary()

#Adding Dense Layers

#So far, we have just completed the convolutional base. Now we need to take these extracted features and add a way to classify them.
#This is why we add the following layers to our model.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#Let's have a look at our model so far
model.summary()

#Training

#Now we will train and compile the model using the recommended hyper paramaters from tensorflow.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels))

#Evaluating the Model

#We can determine how well the model performed by looking at it's performance on the test data set.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

#Working with Small Datasets

#In the situation where you don't have millions of images it is difficult to train a CNN from scratch that performs very well.
#This is why we will learn about a few techniques we can use to train CNN's on small datasets of just a few thousand images. 

#Data Augmentation

#To avoid overfitting and create a larger dataset from a smaller one we can use a technique called data augmentation.
#This is simply performing random transofrmations on our images so that our model can generalize better.
#These transformations can be things like compressions, rotations, stretches and even color changes. 

#Fortunately, keras can help us do this. Look at the code below to an example of data augmentation.
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#creates a data generator object that transforms images
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

#pick an image to transform
test_img = train_images[20]
img = image.img_to_array(test_img)  #convert image to numpy arry
img = img.reshape((1,) + img.shape)  #reshape image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  #this loops runs forever until we break, saving images to current directory with specified prefix
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  #show 4 images
        break

plt.show()