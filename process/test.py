import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

common_voice = load_dataset("common_voice", "en", split="validation", streaming=True)

inputs = feature_extractor(next(iter(common_voice))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

print("Environment set up successful?", logits.shape[-1] == 51865)
