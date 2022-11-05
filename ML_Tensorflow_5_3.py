#Natural Language Processing

#RNN Play Generator

#Imports and Setup
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

#Dataset

#For this example, we only need one peice of training data.
#In fact, we can write our own poem or play and pass that to the network for training if we'd like.
#However, to make things easy we'll use an extract from a shakesphere play.
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#Loading Your Own Data

#To load your own data, you'll need to upload a file from the dialog below.
#Then you'll need to follow the steps from above but load in this new file instead.
from google.colab import files
path_to_file = list(files.upload().keys())[0]

#Read Contents of File

#Let's look at the contents of the file.
#Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#Length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

#Take a look at the first 250 characters in text
print(text[:250])

#Encoding

#Since this text isn't encoded yet well need to do that ourselves. We are going to encode each unique character as a different integer.
vocab = sorted(set(text))
#Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

#Lets look at how part of our text is encoded
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

#And here we will make a function that can convert our numeric values to text.
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))

#Creating Training Examples

#Remember our task is to feed the model a sequence and have it return to us the next character.
#This means we need to split our text data from above into many shorter sequences that we can pass to the model as training examples. 
#Length of sequence for a training example
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

#Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#Next we can use the batch method to turn this stream of characters into batches of desired length.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

#Now we need to use these sequences of length 101 and split them into input and output.
def split_input_target(chunk):  #For the example: hello
    input_text = chunk[:-1]  #hell
    target_text = chunk[1:]  #ello
    return input_text, target_text  #hell, ello

dataset = sequences.map(split_input_target)  #We use map to apply the above function to every entry

for x, y in dataset.take(2):
  print("\n\nEXAMPLE\n")
  print("INPUT")
  print(int_to_text(x))
  print("\nOUTPUT")
  print(int_to_text(y))

#Finally we need to make training batches.
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  #Vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

#Buffer size to shuffle the dataset
#(TF data is designed to work with possibly infinite sequences,
#so it doesn't attempt to shuffle the entire sequence in memory.
#Instead, it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#Building the Model

#Now it is time to build the model.
#We will use an embedding layer a LSTM and one dense layer that contains a node for each unique character in our training data.
#The dense layer will give us a probability distribution over all nodes.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

#Creating a Loss Function

#Now we are going to create our own loss function for this problem.
#This is because our model will output a (64, sequence_length, 65) shaped tensor that represents
#the probability distribution of each character at each timestep for every sequence in the batch.
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  #Ask our model for a prediction on our first batch of training data (64 entries)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  #Print out the output shape

#We can see that the predicition is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)

#Lets examine one prediction
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
#Notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step

#And finally well look at a prediction at the first timestep
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
#And of course its 65 values representing the probabillity of each character occuring next

#If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
sampled_indices = tf.random.categorical(pred, num_samples=1)

#Now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars  #And this is what the model predicted for training sequence 1

#So now we need to create a loss function that can compare that output to the expected output and give us some numeric value representing how close the two were.
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#Compiling the Model

#At this point we can think of our problem as a classification problem where the model predicts the probabillity of each unique letter coming next.
model.compile(optimizer='adam', loss=loss)

#Creating Checkpoints

#Now we are going to setup and configure our model to save checkpoinst as it trains.
#This will allow us to load our model from a checkpoint and continue training it.
#Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
#Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#Training

#Finally, we will start training the model.
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])

#Loading the Model

#We'll rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one peice of text to the model and have it make a prediction.
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

#Once the model is finished training, we can find the lastest checkpoint that stores the models weights using the following line.
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

#We can load any checkpoint we want by specifying the exact file to load.
checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))

#Generating Text

#Now we can use the lovely function provided by tensorflow to generate some text using any starting string we'd like.
def generate_text(model, start_string):
  #Evaluation step (generating text using the learned model)

  #Number of characters to generate
  num_generate = 800

  #Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  #Empty string to store our results
  text_generated = []

  #Low temperatures results in more predictable text.
  #Higher temperatures results in more surprising text.
  #Experiment to find the best setting.
  temperature = 1.0

  #Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      #Remove the batch dimension
    
      predictions = tf.squeeze(predictions, 0)

      #Using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      #We pass the predicted character as the next input to the model
      #along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))
