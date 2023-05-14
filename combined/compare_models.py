from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from acoustic_features_pipeline import getAcousticData
from acoustic_features_pipeline import getAcousticModel

from text_features_pipeline import getTextData, getTextModel
from combined_model import getCombinedModel
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

# Text Model
x_text_test, x_text_train, embedding_matrix, vocab_length = getTextData(x_train, x_test)
textModel = getTextModel(x_text_train, y_train, embedding_matrix, vocab_length)
# scoreTextModel = textModel.evaluate(x_text_test, y_test, verbose=1)
scoreTextModel = textModel.predict(x_text_test)

# https://www.tensorflow.org/guide/keras/save_and_serialize
# Acoustic Model
x_acoustic_train, x_acoustic_test = getAcousticData(x_train, x_test)
acousticModel = getAcousticModel(x_acoustic_train, y_train)
# scoreAcousticModel = acousticModel.evaluate(x_acoustic_test, y_test, verbose=1)
scoreAcousticModel = acousticModel.predict(x_acoustic_test)

# Combined Model
combinedModel = getCombinedModel(
    vocab_length, embedding_matrix, x_text_train, x_acoustic_train, y, y_train
)
# scoreCombinedModel = combinedModel.evaluate(
#     [x_text_test, x_acoustic_test], np.array(y_test), verbose=1
# )
combinedModelPrediction = combinedModel.predict([x_text_test, x_acoustic_test])

print("Predict: ")
print("==============================================================================")
y_prediction = np.argmax(combinedModelPrediction, axis=1)
y_test = np.argmax(y_test, axis=1)

labels = df["emotion"].astype(str).unique()
print(labels)
print(y_prediction)
print(y_test)
matrix = confusion_matrix(y_test, y_prediction, normalize="pred")

df_cm = pd.DataFrame(matrix, labels, labels)
# plt.figure(figsize=(20, 5))
# plt.subplot(1, 3, 1)
sn.heatmap(df_cm, annot=True, cmap="BuPu")  # font size
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Combined Model")
plt.tight_layout()
plt.show()


# print("Evaluate the models: Returns the loss value & metrics values for the model in test mode.")
# print("=========================================================================================")
# print("Text Model Test Score:", scoreTextModel[0])
# print("Text Model Test Accuracy:", scoreTextModel[1])
# print("Acoustic Model Test Score:", scoreAcousticModel[0])
# print("Acoustic Model Test Accuracy:", scoreAcousticModel[1])
# print("Combined Model Test Score:", scoreCombinedModel[0])
# print("Combined Model Test Accuracy:", scoreCombinedModel[1])
