import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import filedialog
import numpy as np
import json
import pickle

max_sequence_length = 1000
root = tk.Tk()
root.withdraw()
progress_file_path = filedialog.askopenfilename()

with open(progress_file_path, 'r') as file:
    logs = file.readlines()

tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(logs)
sequences = tokenizer.texts_to_sequences(logs)
vocab_size = len(tokenizer.word_index) + 1

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

labels = np.array([1] * len(padded_sequences))

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(padded_sequences, labels, epochs=10, batch_size=32)

model.save('log_analysis_model.h5')
