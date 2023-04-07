import pandas as pd
from pathlib import Path
from datasets import load_dataset, Audio
from scipy.io.wavfile import read
import numpy as np
from datasets import Dataset
import json

# Merge all three datasets into one CSV file
df_en = pd.read_csv("data/transcripts/transcriptions_en.csv", sep=";")
df_en = df_en.assign(audio=lambda x: "data/wav_corpus/" + x.file_name + ".wav")
df_en = df_en.drop(columns=["file_name"])
df_en = df_en.assign(language="en")
df_en = df_en.rename(
    columns={
        "manual transcription": "transcription"
    }
)

df_es = pd.read_csv("data/transcripts/transcriptions_es.csv", sep=";")
df_es = df_es.assign(audio=lambda x: "data/wav_corpus/" + x.file_name + ".wav")
df_es = df_es.drop(columns=["file_name"])
df_es = df_es.assign(language="es")
df_es = df_es.rename(
    columns={
         "manual transcription": "transcription"
            }
)

df_it = pd.read_csv("data/transcripts/transcriptions_it.csv", sep=";")
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
df.to_csv(Path("data/transcriptions.csv"), index=False, header=True, sep=";")


# # remove rows that contain unusable information
# import re
# chars_to_remove_regex = '[\=\~\@\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

# def remove_special_characters(batch):
#     batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
#     return batch

# atcosim = pd.read_csv("data/newdata.csv")
# atcosim = atcosim[atcosim["transcription"].str.contains("<OT>|<FL>|[EMPTY]|[FRAGMENT]|[HNOISE]|[NONSENSE]|[UNKNOWN]") == False]
# atcosim.set_index('recording_id')
# atcosim_clean = Dataset.from_pandas(atcosim)
# atcosim_clean = atcosim_clean.map(remove_special_characters)
# atcosim_clean = atcosim_clean.remove_columns('__index_level_0__')
# print(atcosim_clean)
# atcosim_clean.to_csv("data/pruneddata.csv", index = False, header=True)


########################## dataset split
# atcosim = load_dataset('csv', data_files='data/pruneddata.csv', split='train')
# atcosim_clean = atcosim.train_test_split(train_size=0.9, seed=42)
# atcosim_main = atcosim_clean['train'].train_test_split(train_size=0.89, seed=42)
# atcosim_main["validation"] = atcosim_clean["test"]

# atcosim = DatasetDict({
#     'train': atcosim_main['train'],
#     'test': atcosim_main['test'],
#     'valid': atcosim_main['validation']})
# print(atcosim)
# atcosim.save_to_disk("atcosim_pruned")

########################## Vocabulary builder
# # make a vocabulary
# def extract_all_chars(batch):
#   all_text = " ".join(batch["transcription"])
#   vocab = list(set(all_text))
#   return {"vocab": [vocab], "all_text": [all_text]}


# vocab = atcosim.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=atcosim.column_names)
# vocab_list = list(set(vocab["vocab"][0]))
# vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
# vocab_dict["|"] = vocab_dict[" "]
# del vocab_dict[" "]
# vocab_dict["[UNK]"] = len(vocab_dict)
# vocab_dict["[PAD]"] = len(vocab_dict)
# with open('vocab.json', 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)
