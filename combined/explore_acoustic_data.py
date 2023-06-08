import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def load_wav(vid_path, sr=48000):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr

    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0] : sliced[1]])
    wav_output = np.array(wav_output)
    return wav_output


waveform, sample_rate = librosa.load("../data/wav_corpus/f_gio125aen.wav", sr=48000)
mfcc_features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40).T

# f_gio125aen.wav
# Plot the MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_features, x_axis="time")

plt.show()
# from playsound import playsound
# import wavio

# wavio.write(
#     "myfile.wav", load_wav("../data/wav_corpus/f_ans001aen.wav"), 48000, sampwidth=2
# )
# playsound()
# print("playing sound using  playsound")

# for playing note.wav file
# playsound("../data/wav_corpus/f_ans001aen.wav")

# print("playing sound using  playsound")
