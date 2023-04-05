import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd
from datasets import Audio, load_dataset

from model import Wav2Vec2ForSpeechClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(
        speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    inputs = {key: inputs[key].to(device) for key in inputs}
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [
        {"Emotion": config.id2label[i], "Score": f"{np.round(score * 100, 3)}"}
        for i, score in enumerate(scores)
    ]
    return outputs


emo = load_dataset("csv", data_files="data/transcriptions.csv", split="train", sep=";")
# emo = emo.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = 16000
data_en = emo.filter(lambda x: x['language']== 'en')
path = "data/wav_corpus/" + str(data_en["audio"][2])
outputs = predict(path, sampling_rate)
print(outputs)


# import required module
from playsound import playsound
 
# for playing note.wav file
playsound("data/wav_corpus/" + str(data_en["audio"][2]))
print('playing sound using  playsound')