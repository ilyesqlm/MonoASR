import os
import re
import random
import torch
import numpy as np
import pandas as pd
import torchaudio
from IPython.display import Audio, display, HTML
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel, Features, Audio, Value
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoProcessor, TrainingArguments, Trainer
from evaluate import load
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

torch.cuda.empty_cache()
#from huggingface_hub import login

#login(token="")

"""**2.  Prepare Data, Tokenizer, Feature Extractor**"""

## We used datasets that are hosted on my Hugging Face Hub. In any case, these are the same datasets described in the paper, 
## with only minor adjustments (e.g., column names, shuffling)) made to facilitate code implementation
dataset_name = ""
dataset = load_dataset(dataset_name, trust_remote_code=True)

def convert_audio_format(example):
    audio_data = example["audio"]  
    array = np.array(audio_data["array"], dtype=np.float32)  
    sampling_rate = audio_data["sampling_rate"]

    if sampling_rate != 16000:
        array = torchaudio.functional.resample(
            torch.tensor(array), orig_freq=sampling_rate, new_freq=16000
        )
        sampling_rate = 16000  

    return {
        "audio": {
            "array": array.numpy(), 
            "sampling_rate": sampling_rate, 
        }
    }

dataset = dataset.map(convert_audio_format)

dataset = dataset.remove_columns(["Licence"])

chars_to_remove_regex = '[\,\.\;\:\"\“\%\‘\”\�\(\)\_\«\»]'

def remove_special_characters(batch):
    batch["Text"] = re.sub(chars_to_remove_regex, '', batch["Text"]).lower()
    return batch

dataset = dataset.map(remove_special_characters)

def clean_text(batch):
    batch["Text"] = [re.sub(r'[\xa0\u202f]', ' ', text) for text in batch["Text"]]
    return batch

dataset = dataset.map(clean_text, batched=True)


################################################################
## This commented section contains the code that was originally used to extract the vocabulary from a single dataset and build the tokenizer.
# However, in our case, we created a processor that unifies the vocabularies from multiple datasets 
# ("These were then merged into a unified vocabulary, which was used to build the tokenizer and processor applied throughout training").
# So here, we use the processor we created and uploaded to our Hugging Face Hub.
################################################################

'''
#print(dataset["train"][0]["Text"])

#show_random_elements(dataset['train'].remove_columns(["audio"]), num_examples=10)
# Extract all chars

def extract_all_chars(batch):
  all_text = " ".join(batch["Text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = dataset['train'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['train'].column_names)
vocab_test = dataset['test'].map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset['test'].column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
#vocab_dict

"""To make it clearer that " " has its own token class, we give it a more visible character |. In addition, we also add an "unknown" token so that the model can later deal with characters not encountered in Common Voice's training set."""

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

"""Finally, we also add a padding token that corresponds to CTC's "blank token". The "blank token" is a core component of the CTC algorithm. For more information, please take a look at the "Alignment" section here."""

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
#len(vocab_dict)

#vocab_dict

#target_lang = "kab"

#new_vocab_dict = {target_lang: vocab_dict}
new_vocab_dict=vocab_dict
#new_vocab_dict
'''
"""Let's now save the vocabulary as a json file."""

'''
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(new_vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", #target_lang=target_lang
                                                 )

repo_name = ""

# Push tokenizer in the hub
#tokenizer.push_to_hub(repo_name)

""" Create Wav2Vec2FeatureExtractor"""

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

#processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
'''
"""Create the processor"""

repo_name = ""
processor_name = ""
processor = AutoProcessor.from_pretrained(processor_name)

"""**3.  Preprocess Data**"""

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch["Text"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset['train'].column_names)

"""**4.  Set-up Trainer**"""

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
   
processor_name = ""
processor = AutoProcessor.from_pretrained(processor_name)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load("wer")
bleu_metric = load("bleu")
rouge_metric = load("rouge")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
   
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    bleu = {"bleu": 0.0} 
    if all(len(ref) > 0 for ref in label_str):  
        bleu = bleu_metric.compute(predictions=pred_str, references=[[ref] for ref in label_str])
    
    rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0} 
    if all(len(ref) > 0 for ref in label_str):  
        rouge = rouge_metric.compute(predictions=pred_str, references=label_str)
    
    return {
        "wer": wer,
        "bleu": bleu["bleu"],
        "rouge": rouge
    }


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/mms-1b-all",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)

model.init_adapter_layers()

model.freeze_base_model()

adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True

training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=8,
  gradient_accumulation_steps=4,
  evaluation_strategy="epoch",
  save_strategy="epoch", 
  num_train_epochs=100,
  gradient_checkpointing=True,
  fp16=True,
  logging_strategy="epoch",
  learning_rate=1e-3,
  warmup_steps=100,
  save_total_limit=1,
  push_to_hub=True,
  report_to="none",
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss", 
  greater_is_better=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=processor.feature_extractor,
    #processing_class=processor.feature_extractor,
)


trainer.train()


# Adapt
target_lang="kab"
adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
adapter_file = os.path.join(training_args.output_dir, adapter_file)

safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

#trainer.push_to_hub()

