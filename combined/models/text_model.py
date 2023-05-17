import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers import Embedding, LSTM
from keras.layers import LSTM
import nltk
from numpy import asarray
from numpy import zeros


embedding_dimension = 100
embedding_max_length = 100

# get the transcription for the text model and file name for speech model


# --------------------------------------------------------------------------#
# Preprocessing                                                            #
# --------------------------------------------------------------------------#

nltk.download("stopwords")


def preprocess_text(text):
    # Set all words in the sentence to lower case
    sentence = text.lower()

    # Remove punctuations and numbers, this leaves only letters of the english alphabet
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Remove single characters, because we remove all the punctuations, we are sometimes left with a 's' (Mel's)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

    # Remove dubble white spaces, when we remove single characters an extra white space gets left
    sentence = re.sub(r"\s+", " ", sentence)

    # Remove Stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords.words("english")) + r")\b\s*")
    sentence = pattern.sub("", sentence)

    return sentence


def load_text_data(data):
    x = []
    for sentence in data:
        x.append(preprocess_text(sentence))

    return np.array(x)


def getTextData(x_train, x_test):
    # Add all the preprocessed sentences to an array
    x_text_train = load_text_data(x_train["model_transcription"])
    x_text_test = load_text_data(x_test["model_transcription"])

    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(x_text_train)

    x_text_train = word_tokenizer.texts_to_sequences(x_text_train)
    x_text_test = word_tokenizer.texts_to_sequences(x_text_test)

    # Adding 1 to store dimensions for words for which no pretrained word embeddings exist
    vocab_length = len(word_tokenizer.word_index) + 1

    # Set a max length

    x_text_train = pad_sequences(
        x_text_train, padding="post", maxlen=embedding_max_length
    )
    x_text_test = pad_sequences(
        x_text_test, padding="post", maxlen=embedding_max_length
    )

    # Load GloVe word embeddings and create an Embeddings Dictionary
    embeddings_dictionary = dict()
    glove_file = open(
        "../data/pretrained_glove_embedding/glove.6B.100d.txt", encoding="utf8"
    )

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype="float32")
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    # Create an embedding Matrix with 100-dimensional word embeddings
    embedding_matrix = zeros((vocab_length, embedding_dimension))
    for word, index in word_tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return x_text_train, x_text_test, embedding_matrix, vocab_length


# --------------------------------------------------------------------------#
# LSTM                                                                      #
# --------------------------------------------------------------------------#
def getTextModel(x_text_train, y_train, vocab_length, embedding_matrix):
    model = Sequential()
    model.add(
        Embedding(
            vocab_length,
            embedding_dimension,
            weights=[embedding_matrix],
            input_length=embedding_max_length,
            trainable=True,
        )
    )

    model.add(LSTM(256, return_sequences=True, input_shape=(x_text_train.shape[1], 1)))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(256))
    model.add(Dense(y_train.shape[1]))  # for each label
    model.add(Activation("softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
