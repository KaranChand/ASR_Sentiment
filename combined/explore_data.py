import librosa
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import contractions
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

nltk.download("stopwords")
# Read the csv file with data to a data frame
df = pd.read_csv("../output/audio2text/" + "Whisper_english.csv", sep=";")

print(df["emotion"].value_counts())


def preprocess_text(text):
    expanded_words = []
    for word in text.split():
        # using contractions.fix to expand the shortened words
        expanded_words.append(contractions.fix(word))

    sentence = " ".join(expanded_words)
    # Set all words in the sentence to lower case
    sentence = sentence.lower()

    # Remove punctuations and numbers, this leaves only letters of the english alphabet
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Remove single characters, because we remove all the punctuations, we are sometimes left with a 's' (Mel's)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

    # Remove dubble white spaces, when we remove single characters an extra white space gets left
    sentence = re.sub(r"\s+", " ", sentence)

    # Remove Stopwords
    pattern_stopwords = re.compile(
        r"\b(" + r"|".join(stopwords.words("english")) + r")\b\s*"
    )
    sentence = pattern_stopwords.sub("", sentence)

    # Remove the word non-verbal, this occurs when the transcription model can not transcribe the speech segment
    pattern_non_verbal = re.compile("(\s*)non-verbal(\s*)")
    sentence = pattern_non_verbal.sub("", sentence)

    return sentence


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


full_list = []  # list containing all words of all texts
for elmnt in df["model_transcription"]:  # loop over lists in df
    full_list.append(preprocess_text(elmnt))  # append elements of lists to full list

data = get_top_n_words(full_list, 20)

x = [item[0] for item in data]
y = [item[1] for item in data]

plt.figure(figsize=(10, 6))
plt.barh(x, y)  # Use barh for horizontal bars
plt.ylabel("Words")  # Swap x and y labels
plt.xlabel("Frequency")
plt.title("Word Frequency")

plt.tight_layout()
plt.show()


# def load_wav(vid_path, sr=48000):
#     wav, sr_ret = librosa.load(vid_path, sr=sr)
#     assert sr_ret == sr

#     intervals = librosa.effects.split(wav, top_db=20)
#     wav_output = []
#     for sliced in intervals:
#         wav_output.extend(wav[sliced[0] : sliced[1]])
#     wav_output = np.array(wav_output)
#     return wav_output


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
