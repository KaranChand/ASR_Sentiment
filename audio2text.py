from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoProcessor, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset, load_from_disk
import torch

torch.cuda.empty_cache()

for language in ['english', 'italian', 'spanish']:

  # load model and processor
  processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
  forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
  model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
  model.config.forced_decoder_ids = None
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  model.to(device)

  # loading data
  emo = load_dataset('csv', data_files='data/transcriptions/transcriptions.csv', split='train', sep=';')      # for making a full dataset with input values
  emo = emo.filter(lambda x: x["language"] == language)
  emo = emo.cast_column("audio", Audio(sampling_rate=16000))

  def prepare_dataset(x):
    input_features = processor(x['audio']["array"], return_tensors="pt", sampling_rate=x['audio']["sampling_rate"]).to(device).input_features
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    x['model_transcription'] = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return x

  # save
  ds = emo.map(prepare_dataset, remove_columns='audio')
  ds.to_csv("output/Whisper_"+language+".csv", index = False, header=True, sep =';')