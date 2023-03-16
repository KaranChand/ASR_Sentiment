# Project 0D: Sentiment Analysis and Emotion Recognition with ASR.
# Collaborative Pipeline
Models: a model from huggingface  
Data: Ponyland or huggingface  
Model-features:  
sentiment-analysis: output {negative, neutral, positive}

## Mel: What are the most effective techniques for combining automatic speech recognition and sentiment?
Variable dataset. Show different accuracies on different datasets.

## Karan: How robust are sentiment analysis/emotion recognition models to variability in speaking style, accent, and language? 
Variable models. Best techniques to increase accuracy.  
Models: Whisper, Wav2Vec2, Kaldi - huggingface

NOTE from teacher: Some Whisper model does punctuation and capitalization automatically

# TO DO
1. Choose Dataset containing speech and sentiment label -> https://superkogito.github.io/SER-datasets/# and https://huggingface.co/datasets/asapp/slue
2. Choose End2End audio to text transcriber (with punctuation)
3. Choose Sentiment Analysis Model
4. Research acoustic features + word embeddings
