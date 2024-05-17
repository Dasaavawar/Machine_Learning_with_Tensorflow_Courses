# Machine Learning with TensorFlow - Code 1.1. - Introduction to Tensorflow


# TensorFlow 2.0 Introduction

# Installing TensorFlow
# pip install tensorflow
# pip install tensorflow-gpu

# Importing TensorFlow
# %tensorflow_version 2.x  # this line is not required unless you are in a notebook
import tensorflow as tf  # now import the tensorflow module
print(tf.version)  # make sure the version is 2.x

# Tensors
# Creating Tensors
string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# Rank/Degree of Tensors
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

# To determine the rank of a tensor
tf.rank(rank2_tensor)

# Shape of Tensors
rank2_tensor.shape

# Changing Shape
tensor1 = tf.ones([1,2,3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
# Which means this will reshape the tensor to [3,3]
# The numer of elements in the reshaped tensor MUST match the number in the original

# Now let's have a look at our different tensors.
print(tensor1)
print(tensor2)
print(tensor3)
# Notice the changes in shape

# Slicing Tensors
# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) 
print(tf.rank(tensor))
print(tensor.shape)

# Now lets select some different rows and columns from our tensor
three = tensor[0,2]  # selects the 3rd element from the 1st row
print(three)  # -> 3

row1 = tensor[0]  # selects the first row
print(row1)

column1 = tensor[:, 0]  # selects the first column
print(column1)

row_2_and_4 = tensor[1::2]  # selects second and fourth row
print(row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3)
