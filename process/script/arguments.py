from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, 
        metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, 
        metadata={
            "help": "Whether to freeze the entire encoder of the seq2seq model."
            }
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, 
        metadata={
            "help": "A list of tokens that will be suppressed at generation."
        }
    )
    model_index_name: str = field(
        default=None, 
        metadata={"help": "Pretty name for the model card."}
    )
    model_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "The dropout probability to use in the model."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    do_remove_punctuation: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be striped of punctuation."},
    )
    do_normalize_eval: bool = field(
        default=True,
        metadata={"help": "Whether to normalise the references and predictions in the eval WER calculation."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "The number of streamed examples to download before shuffling them. The large the buffer, "
                "the closer it is to real offline shuffling."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use streaming mode to load and pre-process the data."},
    )




