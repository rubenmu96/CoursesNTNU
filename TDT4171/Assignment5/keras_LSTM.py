import pickle
import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, LSTM
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

# Load and prepare data
with open(file="keras-data.pickle", mode="rb") as file:
    keras_data = pickle.load(file)
    
x_train = keras_data['x_train']
y_train = np.array(keras_data['y_train'])
x_test = keras_data['x_test']
y_test = np.array(keras_data['y_test'])
vocab = keras_data['vocab_size'] # number of unique words
max_length = keras_data['max_length']

print('The max lenght is:', max_length)
print('The amount of unique words:', vocab)

# Make train and test data the same length
length = 128
padded_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, 
    maxlen= length,
    # maxlen = max_length, 
    padding = 'post'
)
padded_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test, 
    maxlen= length,
    # max_len = max_length, 
    padding = 'post'
)

# Use a seed to get reproducible results
tf.random.set_seed(1)

# Create a network 
model = Sequential()
model.add(Embedding(vocab, 64, input_length=length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3, seed = 1))
model.add(LSTM(128 , return_sequences=True))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.3, seed = 1))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

tf.debugging.set_log_device_placement(True)

# We use Adam optimizer with Amsgrad
opt = tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad = True)

model.compile(optimizer=opt,
              loss='binary_crossentropy', 
              metrics=['accuracy'])
# Train model
with tf.device("/device:GPU:0"):
    history = model.fit(padded_train, y_train, validation_data=(padded_test, y_test), epochs = 10, batch_size = 128)

# Plot the accuracy and loss
#  "Accuracy"
plt.figure(figsize = (13,6))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# "Loss"
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()