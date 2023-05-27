import librosa
import numpy as np
import pandas as pd

# Read the csv file with data to a data frame
df = pd.read_csv("../output/audio2text/" + "Whisper_english.csv", sep=";")

print(df["emotion"].value_counts())


def load_wav(vid_path, sr=16000):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr

    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0] : sliced[1]])
    wav_output = np.array(wav_output)
    return wav_output


from playsound import playsound
import wavio

wavio.write(
    "myfile.wav", load_wav("../data/wav_corpus/f_ans001aen.wav"), 16000, sampwidth=2
)
# playsound()
# print("playing sound using  playsound")

# for playing note.wav file
# playsound("../data/wav_corpus/f_ans001aen.wav")

# print("playing sound using  playsound")
