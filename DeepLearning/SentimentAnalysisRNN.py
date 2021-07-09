# check from user reviews whether user like the movie or not

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

# preparing data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=25000)

# limiting the content to 100 words
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

# Creating the RNN
model = Sequential()
## Embedding the 25000 unique words into 128 neurons
model.add(Embedding(25000, 128))
## Setting the LSTM layer for RNN   
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4))
## Adding the output layer
model.add(Dense(1, activation='sigmoid'))

# compiling out model and setting up optmizers etc.
model.compile(loss='binary_crossentropy', 
            optimizer='adam',
            metrics=['accuracy'])

# fitting our model in the training data
model.fit(x_train, y_train, batch_size=35, epochs=20, validation_data=(x_test, y_test))

# evaluating out model on our test data
score = model.evaluate(x_test, y_test, batch_size=35)

print('Test score: ', score[0])
print('Test Accuracy: ', score[1]) # 82.30% accuracy