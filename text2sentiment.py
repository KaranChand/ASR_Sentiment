from transformers import pipeline
import pandas as pd
import numpy as np
from pathlib import Path

top_k=None # return all emotions if None, else: number of classes returned
classifier = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", top_k=top_k)

for language in ['english','italian','spanish']:
    filename = 'Whisper_'+language+'.csv'
    transcriptions = pd.read_csv("output/audio2text/"+filename, sep=";")

    model_emotion = np.empty(transcriptions.shape[0], dtype=object)
    for i, transcription in enumerate(transcriptions['model_transcription']):
        emotion_array = classifier(transcription)[0]
        final_emotion_dict = dict()
        for emotion_dict in emotion_array:
            if emotion_dict['label'] in ["positive", "negative"]:
                final_emotion_dict[emotion_dict['label']] = emotion_dict['score']
        model_emotion[i] = max(final_emotion_dict, key=final_emotion_dict.get)
    
    transcriptions['model_emotion'] = model_emotion
    mapping = {
    "fear":"negative",
    "disgust":"negative",
    "happiness":"positive",
    "anger":"negative",
    "sadness":"negative",
    }
    transcriptions = transcriptions.replace({"emotion": mapping})
    transcriptions.to_csv(Path("output/text2sentiment/transcription_sentiment_"+language+".csv"), index=False, header=True, sep=";")