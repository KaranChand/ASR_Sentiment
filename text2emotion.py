from transformers import pipeline
import pandas as pd
import numpy as np
from pathlib import Path

top_k=1 # return all emotions if None, else: number of classes returned
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

for language in ['english','italian','spanish']:
    filename = 'Whisper_'+language+'.csv'
    transcriptions = pd.read_csv("output/audio2text/"+filename, sep=";")

    model_emotion = np.empty(transcriptions.shape[0], dtype=object)
    for i, transcription in enumerate(transcriptions['model_transcription']):
        emotion_array = classifier(transcription)[0]
        final_emotion_dict = dict()
        for emotion_dict in emotion_array:
            if emotion_dict['label'] in ["anger", "joy", "fear", "disgust", "sadness"]:
                final_emotion_dict[emotion_dict['label']] = emotion_dict['score']
        if top_k == None:
            model_emotion[i] = final_emotion_dict
        else:
            model_emotion[i] = max(final_emotion_dict, key=final_emotion_dict.get)
    
    transcriptions['model_emotion'] = model_emotion
    mapping = {"joy": "happiness"}
    transcriptions = transcriptions.replace({"model_emotion": mapping})
    transcriptions.to_csv(Path("output/text2emotion/transcription_emotion_"+language+".csv"), index=False, header=True, sep=";")
