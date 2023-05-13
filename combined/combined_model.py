import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
import keras
from keras.layers.core import Activation, Dropout, Dense

from keras.layers import LSTM, Embedding, LSTM, Flatten

import nltk

from acoustic_features_pipeline import getAcousticData, getAcousticModel

from text_features_pipeline import getTextData, getTextModel


# --------------------------------------------------------------------------#
# Data                                                                      #
# --------------------------------------------------------------------------#
embedding_max_length = 100
embedding_dimension = 100


def getCombinedModel(
    vocab_length, embedding_matrix, x_text_train, x_acoustic_train, y, y_train
):
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
    textModel.add(
        LSTM(256, return_sequences=True, input_shape=(x_text_train.shape[1], 1))
    )
    textModel.add(LSTM(256, return_sequences=False))
    textModel.add(Dense(256))

    # Speech model
    acousticModel = Sequential()
    acousticModel.add(Flatten(input_shape=(x_acoustic_train.shape[1], 1)))
    acousticModel.add(Dense(1024))
    acousticModel.add(Activation("relu"))
    acousticModel.add(Dropout(0.2))
    acousticModel.add(Dense(256))

    # Concatonate the text and speech model
    x = keras.layers.add([textModel.output, acousticModel.output])

    # Concatonate the text and speech model
    x = keras.layers.add([textModel.output, acousticModel.output])

    combinedModel = Sequential()

    combinedModel.add(Activation("relu"))

    combinedModel.add(Dense(256))
    combinedModel.add(Activation("relu"))

    combinedModel.add(Dense(y.shape[1]))
    combinedModel.add(Activation("softmax"))
    combinedModel_output = combinedModel(x)

    model = keras.Model([textModel.input, acousticModel.input], combinedModel_output)

    model.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["categorical_accuracy"]
    )
    model.summary()

    model.fit(
        [x_text_train, x_acoustic_train],
        np.array(y_train),
        batch_size=128,
        epochs=6,
        verbose=1,
    )

    return model
