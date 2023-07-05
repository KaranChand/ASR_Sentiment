from datasets import load_dataset, load_metric, Dataset
import pandas as pd
from pathlib import Path
import csv
import numpy as np
import os
import matplotlib.pyplot as plt

#################### confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

fontsize= 18

plt.figure(figsize=(20, 5))
for i, language in enumerate(["english", "italian", "spanish"]):
    df = pd.read_csv(
        f"output/text2emotion/transcription_emotion_{language}.csv", sep=";"
    )
    y_pred = df["model_emotion_nowhisper"]
    y_true = df["emotion"]
    acc = accuracy_score(y_true, y_pred)
    labels = df["emotion"].unique()
    array = confusion_matrix(y_true, y_pred, labels=labels)

    df_cm = pd.DataFrame(array, labels, labels)
    plt.subplot(1, 3, i + 1)
    sn.heatmap(df_cm, annot=True, cmap="BuPu", fmt='g')  # font size
    plt.ylabel("True", fontsize=fontsize)
    plt.xlabel("Predicted", fontsize=fontsize)
    plt.title(f"{language} with accuracy of {acc*100:.0f}%", fontsize=fontsize+2)
plt.suptitle("Emotion Recognition on Real Transcriptions", fontsize=fontsize+4)
plt.tight_layout()
plt.savefig(f"output/images/emotion_nowhisper.png")
plt.show()


plt.figure(figsize=(20, 5))
for i, language in enumerate(["english", "italian", "spanish"]):
    df = pd.read_csv(
        f"output/text2sentiment/transcription_sentiment_{language}.csv", sep=";"
    )
    y_pred = df["model_emotion_nowhisper"]
    y_true = df["emotion"]
    acc = accuracy_score(y_true, y_pred)
    labels = df["emotion"].unique()
    array = confusion_matrix(y_true, y_pred, labels=labels)

    df_cm = pd.DataFrame(array, labels, labels)
    plt.subplot(1, 3, i + 1)
    sn.heatmap(df_cm, annot=True, cmap="BuPu", fmt='g')  # font size
    plt.ylabel("True", fontsize=fontsize)
    plt.xlabel("Predicted", fontsize=fontsize)
    plt.title(f"{language} with accuracy of {acc*100:.0f}%", fontsize=fontsize+2)
plt.suptitle("Sentiment Analysis on Real Transcriptions", fontsize=fontsize+4)
plt.tight_layout()
plt.savefig(f"output/images/sentiment_nowhisper.png")
plt.show()

# evaluate model predictions using Whisper
plt.figure(figsize=(20, 5))
for i, language in enumerate(["english", "italian", "spanish"]):
    df = pd.read_csv(
        f"output/text2emotion/transcription_emotion_{language}.csv", sep=";"
    )
    y_pred = df["model_emotion"]
    y_true = df["emotion"]
    acc = accuracy_score(y_true, y_pred)
    labels = df["emotion"].unique()
    array = confusion_matrix(y_true, y_pred, labels=labels)

    df_cm = pd.DataFrame(array, labels, labels)
    plt.subplot(1, 3, i + 1)
    sn.heatmap(df_cm, annot=True, cmap="BuPu", fmt='g')  # font size
    plt.ylabel("True", fontsize=fontsize)
    plt.xlabel("Predicted", fontsize=fontsize)
    plt.title(f"{language} with accuracy of {acc*100:.0f}%", fontsize=fontsize+2)
plt.suptitle("Emotion Recognition using Whisper", fontsize=fontsize+4)
plt.tight_layout()
plt.savefig(f"output/images/emotion.png")
plt.show()


plt.figure(figsize=(20, 5))
for i, language in enumerate(["english", "italian", "spanish"]):
    df = pd.read_csv(
        f"output/text2sentiment/transcription_sentiment_{language}.csv", sep=";"
    )
    y_pred = df["model_emotion"]
    y_true = df["emotion"]
    acc = accuracy_score(y_true, y_pred)
    labels = df["emotion"].unique()
    array = confusion_matrix(y_true, y_pred, labels=labels)

    df_cm = pd.DataFrame(array, labels, labels)
    plt.subplot(1, 3, i + 1)
    sn.heatmap(df_cm, annot=True, cmap="BuPu", fmt='g')  # font size
    plt.ylabel("True", fontsize=fontsize)
    plt.xlabel("Predicted", fontsize=fontsize)
    plt.title(f"{language} with accuracy of {acc*100:.0f}%", fontsize=fontsize+2)
plt.suptitle("Sentiment Analysis using Whisper", fontsize=fontsize+4)
plt.tight_layout()
plt.savefig(f"output/images/sentiment.png")
plt.show()

from evaluate import load
cer = load("cer")
wer = load("wer")

for i, language in enumerate(["english", "italian", "spanish"]):
    df = pd.read_csv(
        f"output/audio2text/Whisper_{language}.csv", sep=";"
    )
    y_pred = df["model_transcription"]
    y_true = df["transcription"]
    cer_score = cer.compute(predictions=y_pred, references=y_true)
    wer_score = wer.compute(predictions=y_pred, references=y_true)
    print(f"{language}, WER:{wer_score}, CER:{cer_score}")
