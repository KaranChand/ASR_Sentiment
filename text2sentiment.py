from transformers import pipeline
import pandas as pd
import numpy as np
from pathlib import Path

top_k=1 # return all emotions if None, else 1
emotion = False # use emotion, else use sentiment
if emotion:
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=top_k)
else:
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", top_k=top_k)

for language in ['english','italian','spanish']:
    filename = 'Whisper_'+language+'.csv'
    transcriptions = pd.read_csv("output/audio2text/"+filename, sep=";")

    model_emotion = np.empty(transcriptions.shape[0], dtype=object)
    for i, transcription in enumerate(transcriptions['model_transcription']):
        model_emotion[i] = classifier(transcription)[0][0].get('label')

    transcriptions['model_emotion'] = model_emotion
    if emotion: 
        transcriptions.to_csv(Path("output/text2emotion/transcription_emotion_"+language+".csv"), index=False, header=True, sep=";")
    else:
        transcriptions.to_csv(Path("output/text2sentiment/transcription_sentiment_"+language+".csv"), index=False, header=True, sep=";")