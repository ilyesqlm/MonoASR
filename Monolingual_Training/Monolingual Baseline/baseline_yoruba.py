from datasets import load_dataset, Audio, DatasetDict
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from evaluate import load
from functools import partial
import numpy as np
from transformers import Wav2Vec2ForCTC, AutoProcessor
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
import os
import re
import torchaudio
import random

#from huggingface_hub import login

#login(token="")

## We used datasets that are hosted on my Hugging Face Hub. In any case, these are the same datasets described in the paper, 
## with only minor adjustments (e.g., column names, shuffling)) made to facilitate code implementation

dataset_name = ""
dataset = load_dataset(dataset_name, trust_remote_code=True)


dataset = dataset.remove_columns(["speaker_id"])


chars_to_remove_regex = '[\,\.\;\:\"\“\%\‘\”\�\(\)\_\«\»]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_remove_regex, '', batch["text"]).lower()
    return batch

dataset = dataset.map(remove_special_characters)

def clean_text(batch):
    batch["text"] = re.sub(r'[\xa0\u202f\u200b]', ' ', batch["text"])
    return batch

dataset = dataset.map(clean_text)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

small_train_test = dataset['train'].train_test_split(train_size=7_599, test_size=1_900, seed=42)
val_test = small_train_test['test'].train_test_split(test_size=0.5, seed=42)
final_dataset = {
        'train': small_train_test['train'],
        'validation': val_test['train'],     
        'test': val_test['test']             
    }

dataset = DatasetDict(final_dataset)

print(dataset)
print("train",len(dataset['train']))
print("validation",len(dataset['validation']))
print("test",len(dataset['test']))

def process_batch(batch, processor):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["text"]).input_ids

    return batch



def prepare_dataset(dataset, processor):
    # Apply `process_batch` on the dataset
    dataset = dataset.map(lambda batch: process_batch(batch, processor),
                          remove_columns=dataset['train'].column_names)
    
    return dataset

## We use the processor we created and uploaded to our Hugging Face Hub, which unifies the vocabularies from multiple datasets.
processor_name = ""
processor = AutoProcessor.from_pretrained(processor_name)

# Data Collator
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
            #label_features,
            padding=self.padding,
            return_tensors="pt",
        )
       
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def compute_metrics(pred, processor, wer_metric, bleu_metric, rouge_metric):
    
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    paires = [(p, l) for p, l in zip(pred_str, label_str) if p.strip() != "" and l.strip() != ""]
    if not paires:
        return {"wer": 1.0, "bleu": 0.0, "rouge": 0.0}  
    
    pred_str, label_str = zip(*paires)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    bleu = bleu_metric.compute(predictions=pred_str, references=label_str)
    rouge = rouge_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer, "bleu": bleu, "rouge": rouge}


def load_model(processor):
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
    return model

def main(repo_name="MMS_yoruba", target_lang = "yoruba", dataset=dataset, processor=processor):

    dataset = prepare_dataset(dataset, processor)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load("wer")
    bleu_metric=load("bleu")
    rouge_metric=load("rouge")

    compute_metrics_with_args = partial(compute_metrics, processor=processor, wer_metric=wer_metric, bleu_metric=bleu_metric, rouge_metric=rouge_metric)
    model = load_model(processor)
    training_args = TrainingArguments(
        output_dir=repo_name,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=100,
        gradient_checkpointing=True,
        fp16=True,
        learning_rate=1e-3,
        warmup_steps=500,
        max_grad_norm=1.0,
        weight_decay=1e-3,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        per_device_eval_batch_size=4,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_with_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=processor.feature_extractor,
        callbacks=[],
    )
    trainer.train()
    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)
    adapter_file = os.path.join(training_args.output_dir, adapter_file)
    safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
    #trainer.push_to_hub()
    print("Final evaluation on the test set")
    metrics = trainer.evaluate(eval_dataset=dataset["test"])
    print(metrics)
if __name__ == "__main__":
    main()