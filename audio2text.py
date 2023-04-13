from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoProcessor, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset, load_from_disk
import torch

torch.cuda.empty_cache()

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model.config.forced_decoder_ids = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# loading data
emo = load_dataset('csv', data_files='data/transcriptions/transcriptions.csv', split='train', sep=';')      # for making a full dataset with input values
emo = emo.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(x):
  input_values = processor(x['audio']["array"], return_tensors="pt", padding=True, sampling_rate=x['audio']["sampling_rate"]).to(device).input_values
  predicted_ids = model.generate(input_values)
  x['model_transcription'] = processor.decode(predicted_ids, skip_special_tokens=True)
  return x

# choose how to save
filename = "test"
atcosim = emo.map(prepare_dataset, remove_columns='audio')
atcosim = atcosim.remove_columns(['input_values'])
atcosim.to_csv("output/"+filename+".csv", index = False, header=True, sep =';')