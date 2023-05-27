from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from models.combined_model import getCombinedModel
from keras.utils import np_utils
from models.text_model import getTextData, getTextModel
from models.acoustic_model import getAcousticData, getAcousticModel
import seaborn as sn

test_size = 0.25
random_state = 42
batch_size = 64
epochs = 20
verbose = 0

# Read the csv file with data to a data frame
df = pd.read_csv("../output/audio2text/" + "Whisper_english.csv", sep=";")

labels = df["emotion"].unique()

# Get the transcription andfor the text model and file name for acoustic model
x = df[["model_transcription", "file_name"]]

# Get the emotion labels
y = df["emotion"]


# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=random_state
)
# stratify=y
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_train = np_utils.to_categorical(y_train)

y_test = encoder.transform(y_test)
y_test = np_utils.to_categorical(y_test)

# https://www.tensorflow.org/guide/keras/save_and_serialize

x_text_train, x_text_test, embedding_matrix, vocab_length = getTextData(x_train, x_test)
x_acoustic_train, x_acoustic_test = getAcousticData(x_train, x_test)

text_model = getTextModel(labels, vocab_length, embedding_matrix)
text_model.fit(
    x_text_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
)
text_model_prediction = text_model.predict(x_text_test)


acoustic_model = getAcousticModel(x_acoustic_train, labels)
acoustic_model.fit(
    x_acoustic_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
)
acoustic_model_prediction = acoustic_model.predict(x_acoustic_test)

# Combined Model
combined_model = getCombinedModel(
    x_acoustic_train, labels, vocab_length, embedding_matrix
)

combined_model.fit(
    [x_text_train, x_acoustic_train],
    np.array(y_train),
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
)

combined_model_prediction = combined_model.predict([x_text_test, x_acoustic_test])

text_model_prediction = np.argmax(text_model_prediction, axis=1)
acoustic_model_prediction = np.argmax(acoustic_model_prediction, axis=1)
combined_model_prediction = np.argmax(combined_model_prediction, axis=1)
predictions = [
    text_model_prediction,
    acoustic_model_prediction,
    combined_model_prediction,
]
y_test = np.argmax(y_test, axis=1)


plt.figure(figsize=(20, 5))
for i, prediction in enumerate(predictions):
    labels = df["emotion"].unique()
    array = confusion_matrix(y_test, prediction)

    df_cm = pd.DataFrame(array, labels, labels)
    plt.subplot(1, 3, i + 1)
    sn.heatmap(df_cm, annot=True, cmap="BuPu")  # font size
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(i + 1)
plt.tight_layout()
plt.show()
