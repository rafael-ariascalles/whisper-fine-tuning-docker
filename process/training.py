import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments,TrainerCallback,Seq2SeqTrainer
from datasets import interleave_datasets, load_dataset, IterableDatasetDict, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import subprocess
import evaluate
import logging
import string
import sys
import os
import re

gpu_info = subprocess.run(["nvidia-smi"])
TOKEN = os.getenv("HF_TOKEN")
HUB_MODEL_ID = os.getenv("HF_HUB_MODEL_ID")
PRETRAIN_MODEL_NAME=os.getenv("HF_PRETRAIN_MODEL_NAME")
LANGUAGE=os.getenv("LANGUAGE")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_streaming_dataset(dataset_name, dataset_config_name, split, **kwargs):
    if "+" in split:
        # load multiple splits separated by the `+` symbol *with* streaming mode
        dataset_splits = [load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=True, **kwargs) for split_name in split.split("+")]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=True, **kwargs)
        return dataset


raw_datasets = IterableDatasetDict()

raw_datasets["train"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "es", split="train", use_auth_token=TOKEN)  # set split="train+validation" for low-resource
raw_datasets["test"] = load_streaming_dataset("mozilla-foundation/common_voice_11_0", "es", split="test", use_auth_token=TOKEN)
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16_000))

do_lower_case = True
do_remove_punctuation = False
normalizer = BasicTextNormalizer()
processor = WhisperProcessor.from_pretrained(PRETRAIN_MODEL_NAME, language=LANGUAGE, task="transcribe")

logger.info(raw_datasets["train"].features)

#punctuation_to_remove = string.punctuation.replace("'", "")  # don't remove apostrophes
#punctuation_to_remove_regex = f"[{''.join(punctuation_to_remove)}]"
#if do_remove_punctuation:
#    print("Removing punctuation: ", punctuation_to_remove)

def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=list(next(iter(raw_datasets.values())).features)).with_format("torch")

vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
    buffer_size=800,
    seed=0,
)

##beter exclude model
max_input_length = 30.0
def is_audio_in_length_range(length):
    return length < max_input_length

vectorized_datasets["train"] = vectorized_datasets["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

do_normalize_eval = True

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=do_normalize_eval)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=do_normalize_eval)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

model = WhisperForConditionalGeneration.from_pretrained(PRETRAIN_MODEL_NAME)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
model.config.dropout = 0.15
model.config.attention_dropout = 0.05

num_freezed_params = 180
for i,(name, param) in enumerate(model.named_parameters()):
    if i < num_freezed_params:
        param.requires_grad = False

logging.info("Non-freeze layers {} of {}".format(i-num_freezed_params,i))

training_args = Seq2SeqTrainingArguments(
    output_dir="./{}".format(HUB_MODEL_ID),  # your repo name
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1.5e-4,
    warmup_steps=500,
    max_steps=6000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=64,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=4000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,  # your repo name
    hub_token=TOKEN
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[ShuffleCallback()],
)

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

trainer.train()
#eval_result = trainer.evaluate(eval_dataset=vectorized_datasets["test"])
# save best model, metrics and create model card
trainer.create_model_card(model_name=HUB_MODEL_ID)

kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "language": "es",
    "model_name": "Whisper tiny Spanish - Rjac",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-tiny",
    "tasks": "automatic-speech-recognition",
    "tags": "whisper-event",
}

trainer.push_to_hub(**kwargs)