import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

with open('sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
    text = file.read()



tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1 #one is added because one token is for out of vocab word


input_seq = []

for sen_line in text.split('\n'):
    token_line_list = tokenizer.texts_to_sequences([sen_line])[0]
    for i in range(1, len(token_line_list)):
        n_gram_sequence = token_line_list[:i+1]
        input_seq.append(n_gram_sequence)

maximum_len = 0
for sequence in input_seq:
    if len(sequence) > maximum_len:
        maximum_len = len(sequence)
input_sequences = np.array(pad_sequences(input_seq, maxlen=maximum_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]



y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_words))

model = Sequential()
model.add(Embedding(total_words, 100, input_length=maximum_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))
print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=1)

model.save("next_word_predictor.h5")
print("Model saved as next_word_predictor.h5")


