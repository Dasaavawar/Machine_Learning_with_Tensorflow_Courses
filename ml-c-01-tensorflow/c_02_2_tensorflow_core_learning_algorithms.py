# Machine Learning with TensorFlow - Code 2.2. - TensorFlow Core Learning Algorithms


# Classification

# Imports and Setup
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

# Datasets
# This specific dataset seperates flowers into 3 different classes of species.
# - Setosa
# - Versicolor
# - Virginica

# The information about each flower is the following.
# - sepal length
# - sepal width
# - petal length
# - petal width

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

# Let's have a look at our data.
train.head()

# Now we can pop the species column off and use that as our label.
train_y = train.pop('Species')
test_y = test.pop('Species')
train.head() # the species column is now gone

train.shape  # we have 120 entires with 4 features

# Input Function
def input_fn(features, labels, training=True, batch_size=256):
    # convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Feature Columns
# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

# Building the Model
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # the model must choose between 3 classes.
    n_classes=3)

# Training
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
# we include a lambda to avoid creating an inner function previously

# Evaluation
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Predictions
def input_fn(features, batch_size=256):
    # convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit(): valid = False
  predict[feature] = [float(val)]  

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))

# Here is some example input and expected classes you can try above
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
