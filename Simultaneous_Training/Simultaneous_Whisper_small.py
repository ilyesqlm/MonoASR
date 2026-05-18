import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

"""
    We used datasets and a processor that we created and uploaded to our Hugging Face Hub. 
    The processor unifies the vocabularies from multiple datasets, and the datasets themselves are the same as those described in the paper,  
    with only minor adjustments (e.g., column names, shuffling) made to facilitate code implementation.
"""

model_id = "openai/whisper-small"
processor_name = ""
output_dir = "./whisper_multilingual_small"
yoruba_dataset_name = ""
wolof_dataset_name = ""
french_dataset_name = ""
arabic_dataset_name = ""
kabyle_dataset_name = ""
num_epochs = 100
batch_size = 4
gradient_accumulation_steps = 8
learning_rate = 1e-5
#n_train = 
#n_val = 
#n_test = 
seed = 42

chars_to_remove_regex = r'[\,\.\;\:\"\“\%\‘\”\�\(\)\_\«\»]'

def remove_special_characters(batch):
    batch["Text"] = re.sub(chars_to_remove_regex, '', batch["Text"]).lower()
    return batch

def clean_text(batch):
    batch["Text"] = [re.sub(r'[\xa0\u202f\u200b]', ' ', text) for text in batch["Text"]]
    batch["Text"] = [t.strip() for t in batch["Text"]]
    return batch

def standard_split_from_train(dataset_dict, n_train, n_val, n_test, seed=42):
    train_full = dataset_dict["train"].shuffle(seed=seed)
    assert len(train_full) >= n_train + n_val + n_test, "Dataset"
    train_data = train_full.select(range(n_train))
    val_data = train_full.select(range(n_train, n_train + n_val))
    test_data = train_full.select(range(n_train + n_val, n_train + n_val + n_test))
    return train_data, val_data, test_data

def load_yoruba_dataset(name):
    ds = load_dataset(name, trust_remote_code=True)
    ds = ds.rename_column("text", "Text")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.map(remove_special_characters)
    ds = ds.map(clean_text, batched=True)
    keep_cols = ["audio", "Text"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    return standard_split_from_train(ds, n_train, n_val, n_test, seed)

def load_wolof_dataset(name):
    ds = load_dataset(name, trust_remote_code=True)
    ds = ds.rename_column("transcription", "Text")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.map(remove_special_characters)
    ds = ds.map(clean_text, batched=True)
    keep_cols = ["audio", "Text"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    return standard_split_from_train(ds, n_train, n_val, n_test, seed)

def load_french_dataset(name):
    ds = load_dataset(name, trust_remote_code=True)
    ds = ds.rename_column("sentence", "Text")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.map(remove_special_characters)
    ds = ds.map(clean_text, batched=True)
    keep_cols = ["audio", "Text"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    return standard_split_from_train(ds, n_train, n_val, n_test, seed)

def load_arabic_dataset(name):
    ds = load_dataset(name, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.map(remove_special_characters)
    ds = ds.map(clean_text, batched=True)
    keep_cols = ["audio", "Text"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    return standard_split_from_train(ds, n_train, n_val, n_test, seed)

def load_kabyle_dataset(name):
    ds = load_dataset(name, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.map(remove_special_characters)
    ds = ds.map(clean_text, batched=True)
    keep_cols = ["audio", "Text"]
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])
    return standard_split_from_train(ds, n_train, n_val, n_test, seed)

languages = {}

languages["yoruba"] = load_yoruba_dataset(yoruba_dataset_name)
languages["wolof"] = load_wolof_dataset(wolof_dataset_name)
languages["french"] = load_french_dataset(french_dataset_name)
languages["arabic"] = load_arabic_dataset(arabic_dataset_name)
languages["kabyle"] = load_kabyle_dataset(kabyle_dataset_name)

train_sets = []
val_sets = []
test_sets = []

for lang, (train_ds, val_ds, test_ds) in languages.items():
    train_ds = train_ds.add_column("lang", [lang] * len(train_ds))
    val_ds = val_ds.add_column("lang", [lang] * len(val_ds))
    test_ds = test_ds.add_column("lang", [lang] * len(test_ds))

    languages[lang] = (train_ds, val_ds, test_ds)

    train_sets.append(train_ds)
    val_sets.append(val_ds)
    test_sets.append(test_ds)

train_global = concatenate_datasets(train_sets).shuffle(seed=seed)
val_global = concatenate_datasets(val_sets).shuffle(seed=seed)
test_global = concatenate_datasets(test_sets).shuffle(seed=seed)

dataset = DatasetDict({
    "train": train_global,
    "validation": val_global,
    "test": test_global,
})

print(dataset)
processor = WhisperProcessor.from_pretrained(processor_name)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|pad|>"
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<|endoftext|>"
if tokenizer.bos_token is None:
    tokenizer.bos_token = "<|startoftext|>"

model = WhisperForConditionalGeneration.from_pretrained(model_id, ignore_mismatched_sizes=True)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.resize_token_embeddings(len(tokenizer))

model.generation_config.language = None
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["Text"]).input_ids
    return batch

def map_and_prepare(ds):
    return ds.map(
        prepare_dataset,
        remove_columns=[c for c in ds.column_names if c not in ["audio", "Text", "lang"]]
    )


train_prepared = map_and_prepare(dataset["train"])
val_prepared = map_and_prepare(dataset["validation"])
test_prepared_global = map_and_prepare(dataset["test"])

test_prepared_by_lang = {}
for lang, (_, _, test_ds) in languages.items():
    test_prepared_by_lang[lang] = map_and_prepare(test_ds)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

wer_metric = evaluate.load("wer")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    bleu = bleu_metric.compute(predictions=pred_str, references=[[ref] for ref in label_str])["bleu"]
    
    rouge = rouge_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],     
        "rougeL": rouge["rougeL"],
    }

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=500,
    num_train_epochs=num_epochs,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="epoch",      
    save_strategy="epoch",
    predict_with_generate=True,
    save_total_limit=1,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_prepared,
    eval_dataset=val_prepared,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    callbacks=[],
)

start_time = time.time()
train_start = time.time()

trainer.train()

train_end = time.time()
train_duration = (train_end - train_start) / 60
eval_start = time.time()
metrics = trainer.evaluate(eval_dataset=test_prepared_global)
eval_end = time.time()
eval_duration = (eval_end - eval_start) / 60
total_duration = (eval_end - start_time) / 60

print(metrics)

model.save_pretrained(f"{output_dir}/final_model")
processor.save_pretrained(f"{output_dir}/final_processor")

def evaluate_language(lang, ds):
    pred_ids = trainer.predict(ds).predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = [l for l in ds["labels"]]
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    bleu = bleu_metric.compute(predictions=pred_str, references=[[ref] for ref in label_str])["bleu"]
    rouge = rouge_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
    }

for lang in test_prepared_by_lang:
    print(f"\n====== 🌐 {lang.upper()} ======")
    metrics_lang = evaluate_language(lang, test_prepared_by_lang[lang])
    print(metrics_lang)

