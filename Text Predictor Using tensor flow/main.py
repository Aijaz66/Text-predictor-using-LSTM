import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


trained_tokenizer = Tokenizer()

with open('sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
    text = file.read()
trained_tokenizer.fit_on_texts([text])
total_words = len(trained_tokenizer.word_index) + 1
max_sequence_len = 10

model = load_model("next_word_predictor.h5")
print("Model loaded successfully.")


def predict_next_words(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = trained_tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in trained_tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


seed_text = "you are the one who"
next_words = 7
predicted_text = predict_next_words(seed_text, next_words, max_sequence_len)
print("Predicted Text:", predicted_text)
