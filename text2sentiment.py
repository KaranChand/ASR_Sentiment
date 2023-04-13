from transformers import pipeline
import pandas as pd
import numpy as np
from pathlib import Path

return_all_scores = False
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=return_all_scores)

transcriptions = pd.read_csv("output/test.csv", sep=";")

emotion = np.zeros(transcriptions.shape[0])
for i, transcription in enumerate(transcriptions['model_transcription']):
    emotion[i] = classifier(transcription)

transcriptions['emotion'] = emotion
transcriptions.to_csv(Path("output/transcription_emotion.csv"), index=False, header=True, sep=";")