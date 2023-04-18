from datasets import load_dataset, load_metric, Dataset
import pandas as pd 
from pathlib import Path
import csv
import numpy as np
import os
import matplotlib.pyplot as plt

columns = np.array([])
values = np.array([])
for filename in os.listdir('output'):
    df = pd.read_csv("output/"+filename)
    y_pred = df['y_pred']
    y_test = df['y_test']

    columns = np.append(columns, filename[23:-4])
    values = np.append()

plt.figure(figsize=(8,5))
plt.title("Mean Squared Error of Validation Set")
plt.barh(columns, values)
plt.ylabel("Regressor")
plt.xlabel("y_test - y_pred")
plt.savefig("comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()


# remove rows that have NaN as model_transcription for model evaluation
files = ['transcribed_base_test', 'transcribed_robust_test', 'transcribed_hubert_test', 
'transcribed_xlsr_ft_10_test', 'transcribed_xlsr_ft_50_test','transcribed_xlsr_ft_150_test',
'transcribed_xlsr_ft_500_test', 'transcribed_xlsr_ft_1000_test', "transcribed_xlsr_ft_10_ARPA",
"transcribed_xlsr_ft_50_ARPA", "transcribed_xlsr_ft_150_ARPA", "transcribed_xlsr_ft_5000_ARPA", 
"transcribed_xlsr_ft_1000_ARPA", "transcribed_xlsr_ft_1000_ARPA_train", "transcribed_xlsr_ft_10_ARPA_train",
"transcribed_xlsr_ft_50_ARPA_train", "transcribed_xlsr_ft_150_ARPA_train", "transcribed_xlsr_ft_500_ARPA_train"]

files = ["transcribed_xlsr_ft_500_ARPA_train"]
# compute indices of rows that contain NaN
dirty_indices = set()
for f in files:
        df = pd.read_csv("output/" + f+".csv")
        dirty_indices.update(df.loc[pd.isna(df["model_transcription"]), :].index.values)

# evaluate dataset
ds = load_dataset('csv', data_files='data/pruneddata.csv', split='train')
references = ds['transcription']

# perplexity
perplexity_metric = load_metric("perplexity", module_type="metric")
perplexity = perplexity_metric.compute(model_id='gpt2', input_texts=references)['mean_perplexity']

metrics = {"filename" : 'Atcosim',
                   "perplexity" : perplexity
                }
with open('metrics/data_metrics.csv', 'w') as f:  
    w = csv.DictWriter(f, metrics.keys())
    w.writeheader()
    w.writerow(metrics)

# evaluate transcriptions
for filename in files:
        df = pd.read_csv("output/" + filename+".csv") 
        clean_df = df.drop(dirty_indices)
        clean_df.to_csv(Path("data/clean_newdata.csv"), index = False, header=True)
        print(f"removed {len(dirty_indices)} model_transcriptions that contain NaN")

        # load as dataset
        ds = load_dataset('csv', data_files='data/clean_newdata.csv', split='train')
        predictions = ds['model_transcription']
        references = ds['transcription']

        # word error rate
        wer_metric = load_metric("wer")
        wer = wer_metric.compute(predictions=predictions, references=references)

        # Character error rate
        cer_metric = load_metric("cer")
        cer = cer_metric.compute(predictions=predictions, references=references)
            
        metrics = {"filename" : filename+ '.csv',
                   "wer" : wer,
                   "cer" : cer
                }

        df = pd.read_csv("metrics/transcribed_metrics.csv")
        df.set_index('filename', inplace= True)
        df.loc[filename + '.csv'] = metrics
        df.reset_index(inplace=True)
        df.to_csv(Path("metrics/transcribed_metrics.csv"), index = False, header=True, float_format='%.5f')