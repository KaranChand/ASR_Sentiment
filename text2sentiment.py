from transformers import pipeline
import pandas as pd
import numpy as np
from pathlib import Path

return_all_scores = False
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=return_all_scores)

filename = 'test.csv'
transcriptions = pd.read_csv("output/"+filename, sep=";")

emotion = np.empty(transcriptions.shape[0], dtype=object)
for i, transcription in enumerate(transcriptions['model_transcription']):
    emotion[i] = classifier(transcription)[0].get('label')

transcriptions['emotion'] = emotion
transcriptions.to_csv(Path("output/transcription_emotion.csv"), index=False, header=True, sep=";")