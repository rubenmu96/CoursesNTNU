import pickle
import numpy as np
import tensorflow as tf
import tensorflow
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, LSTM
from tensorflow.keras import Sequential
import keras_tuner as kt
# Load data and process it
with open(file="keras-data.pickle", mode="rb") as file:
    keras_data = pickle.load(file)
    
x_train = keras_data['x_train']
y_train = np.array(keras_data['y_train'])
x_test = keras_data['x_test']
y_test = np.array(keras_data['y_test'])
vocab = keras_data['vocab_size'] 
max_length = keras_data['max_length'] 

# Make pad_sequences to make sure that x_train and x_test have the same length
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

# Use seed to get reproducible results
tf.random.set_seed(1)

def network_tuner(hp):
    # Here we create a network with different possible parameters
    model = Sequential()
    model.add(Embedding(vocab, hp.Choice('units0', [32, 64, 128, 256]), input_length=length))
    model.add(LSTM(hp.Choice('units', [32, 64, 128, 256]),  return_sequences=True)),
    model.add(Dropout(hp.Choice('rate', [0.1, 0.3, 0.5]), seed = 1)),
    model.add(LSTM(hp.Choice('units1', [32, 64, 128, 256]),  return_sequences=True))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(hp.Choice('rate1', [0.1, 0.3, 0.5]), seed = 1))
    model.add(Dense(hp.Choice('units2', [32, 64, 128, 256]), activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(hp.Choice('lr1', [0.01, 0.005, 0.001, 0.0005]), amsgrad = True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

tf.debugging.set_log_device_placement(True)
with tf.device("/device:GPU:0"):
    # Use keras tuner to find good parameters
    tuner = kt.RandomSearch(
        network_tuner,
        objective='val_loss',
        max_trials=20,
        overwrite = True
    )
    tuner.search(padded_train, y_train, epochs=5, batch_size = 128, validation_data=(padded_test, y_test))
    best_model = tuner.get_best_models()[0]
