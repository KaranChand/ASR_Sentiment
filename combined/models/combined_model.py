from keras.models import Sequential
import keras
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import LSTM, Embedding, Flatten

# --------------------------------------------------------------------------#
# Concat the text model and speech model                                    #
# Used some code and inspiration from the following sources:                #
# - Paper: https://arxiv.org/pdf/1804.05788.pdf                             #
# - Code https://github.com/Samarth-Tripathi/IEMOCAP-Emotion-Detection      #
# --------------------------------------------------------------------------#

embedding_max_length = 30  # max words in sentence
embedding_dimension = 300


def getCombinedModel(x_acoustic, labels, vocab_length, embedding_matrix):
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
    textModel.add(LSTM(256, return_sequences=True, recurrent_dropout=0.2))
    textModel.add(Dropout(0.2))
    textModel.add(LSTM(256, return_sequences=False, recurrent_dropout=0.2))
    textModel.add(Dropout(0.2))
    textModel.add(Dense(256))

    # Speech model
    acousticModel = Sequential()
    acousticModel.add(
        LSTM(
            256,
            return_sequences=True,
            input_shape=(x_acoustic.shape[1], 1),
            recurrent_dropout=0.2,
        )
    )
    acousticModel.add(Dropout(0.2))
    acousticModel.add(Flatten())
    acousticModel.add(Dense(256))

    # Concatenate the text and speech model https://github.com/keras-team/keras/issues/3921#issuecomment-335457553
    x = keras.layers.add([textModel.output, acousticModel.output])

    combinedModel = Sequential()

    combinedModel.add(Activation("relu"))

    combinedModel.add(Dense(256))
    combinedModel.add(Activation("relu"))

    combinedModel.add(Dense(labels.shape[0]))
    combinedModel.add(Activation("softmax"))
    combinedModel_output = combinedModel(x)

    model = keras.Model([textModel.input, acousticModel.input], combinedModel_output)

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
