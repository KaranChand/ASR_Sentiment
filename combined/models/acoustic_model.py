import librosa as lb
import librosa.core.pitch as lbcp
from keras.layers import Flatten
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout

# get the transcription for the text model and file name for speech model


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


def getAcousticData(x_train, x_test):
    x_acoustic_train = load_speech_data(x_train["file_name"])
    x_acoustic_test = load_speech_data(x_test["file_name"])
    return x_acoustic_train, x_acoustic_test


# --------------------------------------------------------------------------#
# LSTM                                                                      #
# --------------------------------------------------------------------------#


def getAcousticModel(x_train, y_train):
    model = Sequential()

    model.add(Flatten(input_shape=(x_train.shape[1], 1)))
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dense(256))

    model.add(Dense(y_train.shape[1]))  # 5 for each label
    model.add(Activation("softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adadelta",
        metrics=["accuracy"],
    )

    return model
