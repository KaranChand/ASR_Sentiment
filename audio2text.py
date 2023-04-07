from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoProcessor
from datasets import Audio, load_dataset, load_from_disk
import torch

torch.cuda.empty_cache()

# define pipeline
checkpoint = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
model = AutoModelForCTC.from_pretrained(checkpoint, local_files_only=True)
processor = Wav2Vec2Processor.from_pretrained(checkpoint)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# loading data
emo = load_dataset('csv', data_files='data/transcriptions/transcriptions.csv', split='train', sep=';')      # for making a full dataset with input values
emo = emo.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(x):
  input_values = processor(x['audio']["array"], return_tensors="pt", padding=True, sampling_rate=x['audio']["sampling_rate"]).to(device).input_values
  x['input_values'] = input_values[0]
  logits = model(input_values).logits
  pred_id = torch.argmax(logits, dim=-1)[0]
  x['model_transcription'] = processor.decode(pred_id)
  return x

# choose how to save
filename = "test"
atcosim = emo.map(prepare_dataset, remove_columns='audio')
atcosim = atcosim.remove_columns(['input_values'])
atcosim.to_csv("output/"+filename+".csv", index = False, header=True)