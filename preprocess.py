import pandas as pd
from pathlib import Path
from datasets import load_dataset, Audio
from scipy.io.wavfile import read
import numpy as np
from datasets import Dataset
import json

# Merge all three datasets into one CSV file
df_en = pd.read_csv("data/transcriptions/transcriptions_en.csv", sep=";")
df_en = df_en.assign(audio=lambda x: "data/wav_corpus/" + x.file_name + ".wav")
df_en = df_en.drop(columns=["file_name"])
df_en = df_en.assign(language="en")
df_en = df_en.rename(columns={"manual transcription": "transcription"})

df_es = pd.read_csv("data/transcriptions/transcriptions_es.csv", sep=";")
df_es = df_es.assign(audio=lambda x: "data/wav_corpus/" + x.file_name + ".wav")
df_es = df_es.drop(columns=["file_name"])
df_es = df_es.assign(language="es")
df_es = df_es.rename(columns={"manual transcription": "transcription"})

df_it = pd.read_csv("data/transcriptions/transcriptions_it.csv", sep=";")
df_it = df_it.assign(audio=lambda x: "data/wav_corpus/" + x.file_name + ".wav")
df_it = df_it.drop(columns=["file_name"])
df_it = df_it.assign(language="it")
df_it = df_it.rename(
    columns={
        "manual correction from automatic transcription made with Wav2vec2-large-xlsr-53": "transcription"
    }
)
df = df_en.append(df_es)
df = df.append(df_it)

df = df.assign(emotion=lambda x: x.audio.str[18:21])
mapping = {
    "ans": "fear",
    "dis": "disgust",
    "gio": "happiness",
    "rab": "anger",
    "tri": "sadness",
}
df = df.replace({"emotion": mapping})

mapping = {
    "en": "english",
    "it": "italian",
    "es": "spanish",
}
df = df.replace({"language": mapping})
df.to_csv(
    Path("data/transcriptions/transcriptions.csv"), index=False, header=True, sep=";"
)
