#TensorFlow Core Learning Algorithms

#Linear Regression

#Imports and Setup
import matplotlib.pyplot as plt
import numpy as np

#Linear regression follows a very simple concept. If data points are related linearly,
#we can generate a line of best fit for these points and use it to predict future values.
#Let's take an example of a data set with one feature and one label.
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])

#Here's a refresher on the equation of a line in 2D.
#y=mx+b
#Here's an example of a line of best fit for this graph.
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.show()

#Setup and Imports
#pip install -q sklearn

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

#Data
#The dataset we will be focusing on here is the titanic dataset. It has tons of information about each passanger on the ship.
#Our first step is always to understand the data and explore it. So, let's do that!

#Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#To look at the data we'll use the .head() method from pandas.
#This will show us the first 5 items in our dataframe.
dftrain.head()

#And if we want a more statistical analysis of our data we can use the .describe() method.
dftrain.describe()

#And since we talked so much about shapes in the previous tutorial let's have a look at that too!
dftrain.shape

#Now let's have a look at our survival information.
y_train.head()

#And now because visuals are always valuable let's generate a few graphs of the data.
dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

#Feature Columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  #gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

#Input Function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  #inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  #create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  #randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  #split dataset into batches of 32 and repeat process for number of epochs
    return ds  #return a batch of the dataset
  return input_function  #return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  #here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#Creating the Model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
#We create a linear estimtor by passing the feature columns we created earlier

#Training the Model
linear_est.train(train_input_fn)  #train
result = linear_est.evaluate(eval_input_fn)  #get model metrics/stats by testing on tetsing data

clear_output()  #clears consoke output
print(result['accuracy'])  #the result variable is simply a dict of stats about our model

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')