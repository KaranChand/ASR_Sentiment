import librosa as lb
import librosa.core.pitch as lbcp
from keras.layers import Flatten
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Activation, Dense, Dropout


# --------------------------------------------------------------------------#
# Preproccesing                                                            #
# --------------------------------------------------------------------------#
def load_speech_data(data):
    x = []
    for file_name in data:
        feature = extract_feature(
            "../data/wav_corpus/" + file_name + ".wav",
            extract_mfcc=True,
            extract_chroma=True,
            extract_melspectogram=True,
            magnitudes=True,
            pitches=True,
            extract_contrast=True,
        )
        x.append(feature)

    return np.array(x)


def getAcousticData(x_train, x_test):
    x_acoustic_train = load_speech_data(x_train["file_name"])
    x_acoustic_test = load_speech_data(x_test["file_name"])
    return x_acoustic_train, x_acoustic_test


# --------------------------------------------------------------------------#
# Feature extraction                                                        #
# --------------------------------------------------------------------------#
def extract_feature(
    file_name,
    extract_mfcc,
    extract_chroma,
    extract_melspectogram,
    extract_magnitudes,
    extract_pitches,
    extract_contrast,
):
    sr = 48000
    audio, sample_rate = lb.load(file_name, sr=sr)

    intervals = lb.effects.split(audio, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(audio[sliced[0] : sliced[1]])
    audio = np.array(wav_output)

    pitches, magnitudes = lbcp.piptrack(y=audio, sr=sample_rate)

    stft = np.abs(lb.stft(audio))  # convert a signal into its frequency

    result = np.array([])

    if extract_mfcc:
        mfccs = np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        mfcc_std = np.std(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfcc_std))
    if extract_chroma:
        chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        chroma_std = np.std(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_std))
    if extract_melspectogram:
        mel = np.mean(lb.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        mel_std = np.std(lb.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_std))
    if extract_contrast:
        contrast = np.mean(
            lb.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0
        )
        result = np.hstack((result, contrast))
        contrast_std = np.std(
            lb.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0
        )
        result = np.hstack((result, contrast_std))
    if extract_magnitudes:
        magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]
        result = np.hstack((result, magnitudes))
    if extract_pitches:
        pitches = np.trim_zeros(np.mean(pitches, axis=1))[:20]
        result = np.hstack((result, pitches))

    # todo: librosa noise reduction, silence removal, pre-emphasis from paper: https://link.springer.com/article/10.1007/s11042-020-09874-7#Sec34
    # https://www.kaggle.com/code/jaseemck/audio-processing-using-librosa-for-beginners

    # https://github.com/bagustris/ravdess_song_speech/blob/master/code/extract_librosa_hsf.py
    # https://github.com/RoccoJay/Audio_to_Emotion/blob/master/Audio_to_Emotion.ipynb

    # https://link.springer.com/article/10.1007/s11042-020-09874-7#Sec34

    return result


# --------------------------------------------------------------------------#
# Model                                                                      #
# --------------------------------------------------------------------------#
def getAcousticModel(x_train, labels):
    model = Sequential()

    model.add(
        LSTM(
            256,
            return_sequences=True,
            input_shape=(x_train.shape[1], 1),
            recurrent_dropout=0.2,
        )
    )
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256))

    model.add(Dense(labels.shape[0]))  # 5 for each label
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adadelta",
        metrics=["accuracy"],
    )

    return model
