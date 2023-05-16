from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from models.combined_model import getCombinedModel

from models.text_model import getTextData, getTextModel
from models.acoustic_model import getAcousticData, getAcousticModel
import seaborn as sn

# Read the csv file with data to a data frame
df = pd.read_csv("../output/audio2text/" + "Whisper_english.csv", sep=";")

# Get the transcription andfor the text model and file name for acoustic model
x = df[["model_transcription", "file_name"]]

# Get the emotion labels
y = pd.get_dummies(df["emotion"]).values

# Train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

# https://www.tensorflow.org/guide/keras/save_and_serialize

x_text_train, x_text_test, embedding_matrix, vocab_length = getTextData(x_train, x_test)
x_acoustic_train, x_acoustic_test = getAcousticData(x_train, x_test)

text_model = getTextModel(x_text_train, y_train, vocab_length, embedding_matrix)
text_model.fit(
    x_text_train,
    y_train,
    batch_size=128,
    epochs=3,
    verbose=0,
)
text_model_prediction = text_model.predict(x_text_test)


acoustic_model = getAcousticModel(x_acoustic_train, y_train)
acoustic_model.fit(
    x_acoustic_train,
    y_train,
    batch_size=128,
    epochs=3,
    verbose=0,
)
acoustic_model_prediction = acoustic_model.predict(x_acoustic_test)

# Combined Model
combined_model = getCombinedModel(
    x_text_train, x_acoustic_train, y, vocab_length, embedding_matrix
)

combined_model.fit(
    [x_text_train, x_acoustic_train],
    np.array(y_train),
    batch_size=128,
    epochs=3,
    verbose=0,
)

combined_model_prediction = combined_model.predict([x_text_test, x_acoustic_test])

text_model_prediction = np.argmax(text_model_prediction, axis=1)
acoustic_model_prediction = np.argmax(acoustic_model_prediction, axis=1)
combined_model_prediction = np.argmax(combined_model_prediction, axis=1)
predictions = [
    text_model_prediction,
    acoustic_model_prediction,
    acoustic_model_prediction,
]
y_test = np.argmax(y_test, axis=1)


plt.figure(figsize=(20, 5))
for i, prediction in enumerate(predictions):
    labels = df["emotion"].unique()
    array = confusion_matrix(y_test, prediction, normalize="pred")

    df_cm = pd.DataFrame(array, labels, labels)
    plt.subplot(1, 3, i + 1)
    sn.heatmap(df_cm, annot=True, cmap="BuPu")  # font size
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(i + 1)
plt.tight_layout()
plt.show()


# y_test = np.argmax(y_test, axis=1)

# labels = df["emotion"].astype(str).unique()
# matrix = confusion_matrix(y_test, combined_model_prediction, normalize="pred")

# df_cm = pd.DataFrame(matrix, labels, labels)
# sn.heatmap(df_cm, annot=True, cmap="BuPu")  # font size
# plt.xlabel("True")
# plt.ylabel("Predicted")
# plt.title("Combined Model")
# plt.tight_layout()
# plt.show()
