import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import librosa as lb
import librosa.core.pitch as lbcp
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
import keras
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import (
    Flatten,
    Embedding,
    LSTM,
)
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import nltk
from numpy import asarray
from numpy import zeros


# --------------------------------------------------------------------------#
# Data                                                                      #
# --------------------------------------------------------------------------#

filename = "Whisper_english.csv"
df = pd.read_csv("../output/" + filename, sep=";")

# get the transcription for the text model and file name for speech model
x = df[["model_transcription", "file_name"]]
y = pd.get_dummies(df["emotion"]).values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

# --------------------------------------------------------------------------#
# Text features                                                             #
# Used some code and inspiration from the following sources:                #
# https://github.com/skillcate/sentiment-analysis-with-deep-neural-networks #
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


# Add all the preprocessed sentences to an array
x_text_train = load_text_data(x_train["model_transcription"])
x_text_test = load_text_data(x_test["model_transcription"])

# Convert categorical variable into indicator variables.
y = pd.get_dummies(df["emotion"]).values

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(x_text_train)

x_text_train = word_tokenizer.texts_to_sequences(x_text_train)
x_text_test = word_tokenizer.texts_to_sequences(x_text_test)

# Adding 1 to store dimensions for words for which no pretrained word embeddings exist
vocab_length = len(word_tokenizer.word_index) + 1

# Set a max length
embedding_max_length = 100
x_text_train = pad_sequences(x_text_train, padding="post", maxlen=embedding_max_length)
x_text_test = pad_sequences(x_text_test, padding="post", maxlen=embedding_max_length)

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
embedding_dimension = 100
embedding_matrix = zeros((vocab_length, embedding_dimension))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# --------------------------------------------------------------------------#
# Acoustic features
#
# https://towardsdatascience.com/building-a-speech-emotion-recognizer-using-python-4c1c7c89d713                                                        #
# --------------------------------------------------------------------------#


def extract_feature(file_name, mfcc, chroma, mel):
    audio, sample_rate = lb.load(
        file_name, res_type="kaiser_fast"
    )  # resampy faster method

    pitches, magnitudes = lbcp.piptrack(y=audio, sr=sample_rate)

    if chroma:
        stft = np.abs(lb.stft(audio))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(lb.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

    magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]
    result = np.hstack((result, magnitudes))

    pitches = np.trim_zeros(np.mean(pitches, axis=1))[:20]
    result = np.hstack((result, pitches))
    return result


def load_speech_data(data):
    x = []
    for file_name in data:
        feature = extract_feature(
            "../data/wav_corpus/" + file_name + ".wav", mfcc=True, chroma=True, mel=True
        )
        x.append(feature)

    return np.array(x)


x_speech_train = load_speech_data(x_train["file_name"])
x_speech_test = load_speech_data(x_test["file_name"])
# --------------------------------------------------------------------------#
# Concat the text model and speech model                                    #
# Used some code and inspiration from the following sources:                #
# - Paper: https://arxiv.org/pdf/1804.05788.pdf                             #
# - Code https://github.com/Samarth-Tripathi/IEMOCAP-Emotion-Detection      #
# --------------------------------------------------------------------------#

# text model
textModel = Sequential()
textModel.add(
    Embedding(
        vocab_length,
        embedding_dimension,
        weights=[embedding_matrix],
        input_length=embedding_max_length,
        trainable=True,
    )
)
textModel.add(LSTM(256, return_sequences=True, input_shape=(x_text_train.shape[1], 1)))
textModel.add(LSTM(256, return_sequences=False))
textModel.add(Dense(256))

# Speech model
speechModel = Sequential()
speechModel.add(Flatten(input_shape=(x_speech_train.shape[1], 1)))
speechModel.add(Dense(1024))
speechModel.add(Activation("relu"))
speechModel.add(Dropout(0.2))
speechModel.add(Dense(256))

# Concatonate the text and speech model
x = keras.layers.add([textModel.output, speechModel.output])

combinedModel = Sequential()

combinedModel.add(Activation("relu"))

combinedModel.add(Dense(256))
combinedModel.add(Activation("relu"))

combinedModel.add(Dense(y.shape[1]))
combinedModel.add(Activation("softmax"))
combinedModel_output = combinedModel(x)

model = keras.Model([textModel.input, speechModel.input], combinedModel_output)

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

speechModel.summary()
textModel.summary()
model.summary()


model.fit(
    [x_text_train, x_speech_train],
    np.array(y_train),
    batch_size=32,
    epochs=6,
    verbose=1,
)

print("Shape y train: ", np.array(y_train).shape, "test: ", np.array(y_test).shape)
print("Shape x text train: ", x_text_test.shape, "test: ", x_text_train.shape)
print("Shape x speech train: ", x_speech_test.shape, "test: ", x_speech_train.shape)
score = model.evaluate([x_text_test, x_speech_test], np.array(y_test), verbose=1)

# Model Performance
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
