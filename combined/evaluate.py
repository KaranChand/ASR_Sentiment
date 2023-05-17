import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from models.combined_model import getCombinedModel

from models.text_model import getTextData, getTextModel
from models.acoustic_model import getAcousticData, getAcousticModel

n_splits = 2
random_state = 42
batch_size = 128
epochs = 3
verbose = 0

# Read the csv file with data to a data frame
df = pd.read_csv("../output/audio2text/" + "Whisper_english.csv", sep=";")

# Get the transcription andfor the text model and file name for acoustic model
x = df[["model_transcription", "file_name"]]
# Get the emotion labels

# Get the emotion labels
y = pd.get_dummies(df["emotion"]).values

names = ["Text Model", "Acoustic Model", "Combined Model"]
text_acc_per_fold = []
acoustic_acc_per_fold = []
combined_acc_per_fold = []

cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
for name in names:
    if name == "Text Model":
        for train, test in cv.split(x, y):
            x_text_train, x_text_test, embedding_matrix, vocab_length = getTextData(
                x.iloc[train], x.iloc[test]
            )

            text_model = getTextModel(
                x_text_train, y[train], vocab_length, embedding_matrix
            )

            text_model.fit(
                x_text_train,
                y[train],
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )

            scores = text_model.evaluate(x_text_test, y[test], verbose=0)

            text_acc_per_fold.append(scores[1] * 100)
    elif name == "Acoustic Model":
        for train, test in cv.split(x, y):
            x_acoustic_train, x_acoustic_test = getAcousticData(
                x.iloc[train], x.iloc[test]
            )

            acoustic_model = getAcousticModel(x_acoustic_train, y[train])

            acoustic_model.fit(
                x_acoustic_train,
                y[train],
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )

            scores = acoustic_model.evaluate(x_acoustic_test, y[test], verbose=0)

            acoustic_acc_per_fold.append(scores[1] * 100)

    if name == "Combined Model":
        for train, test in cv.split(x, y):
            x_text_train, x_text_test, embedding_matrix, vocab_length = getTextData(
                x.iloc[train], x.iloc[test]
            )

            x_acoustic_train, x_acoustic_test = getAcousticData(
                x.iloc[train], x.iloc[test]
            )

            combined_model = getCombinedModel(
                x_text_train, x_acoustic_train, y, vocab_length, embedding_matrix
            )

            combined_model.fit(
                [x_text_train, x_acoustic_train],
                np.array(y[train]),
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )

            scores = combined_model.evaluate(
                [x_text_test, x_acoustic_test], np.array(y[test]), verbose=0
            )

            combined_acc_per_fold.append(scores[1] * 100)
print(np.mean(text_acc_per_fold))
print(np.mean(acoustic_acc_per_fold))
print(np.mean(combined_acc_per_fold))


# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
