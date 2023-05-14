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

plt.figure(figsize=(20, 5))
for i, language in enumerate(["english", "italian", "spanish"]):
    df = pd.read_csv(
        f"output/text2emotion/transcription_emotion_{language}.csv", sep=";"
    )
    y_pred = df["model_emotion"]
    y_true = df["emotion"]
    labels = df["emotion"].unique()
    array = confusion_matrix(y_true, y_pred, labels=labels)

    df_cm = pd.DataFrame(array, labels, labels)
    plt.subplot(1, 3, i + 1)
    sn.heatmap(df_cm, annot=True, cmap="BuPu")  # font size
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(language)
plt.tight_layout()
plt.show()
