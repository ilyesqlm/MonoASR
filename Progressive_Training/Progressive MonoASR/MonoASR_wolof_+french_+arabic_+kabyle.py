from evaluate import load
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import load_dataset, Audio, DatasetDict
from tqdm import tqdm
from itertools import cycle
torch.cuda.empty_cache()

from huggingface_hub import login

#login(token="")

class CFG:
    epochs = 100
    batch_size = 4
    gradient_accumulation_steps = 8
    num_workers = 0   
    lr = 1e-3
    """
    We used datasets and a processor that we created and uploaded to our Hugging Face Hub. 
    The processor unifies the vocabularies from multiple datasets, and the datasets themselves are the same as those described in the paper, with only minor adjustments 
    (e.g., column names, shuffling) made to facilitate code implementation.
    """
    processor_name = ""
    wolof_dataset_name = ""
    french_dataset_name = ""
    arabic_dataset_name = ""
    kabyle_dataset_name = ""
    base_model_name = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    language_tokens = 10
    #patience = 15
    weight_decay = 1e-3
    factor = 0.8
    ignore_index = -100

"""**<h1>1. Language Projection Module</h1>**"""

# Normalization
class RMSNorm(nn.Module):
    def __init__(self, dim: int,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def __call__(self, x):
        return self.forward(x)

def default_linear_init(weight):
    nn.init.xavier_uniform_(weight)  

class Attention(nn.Module):
    def __init__(self,
                 lpm_dim:int = 768,
                 n_heads:int = 8,
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.lpm_dim = lpm_dim
        self.head_dim = self.lpm_dim // self.n_heads

        self.wq = nn.Linear(self.lpm_dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.lpm_dim, self.n_heads * self.head_dim, bias=False)   
        self.wv = nn.Linear(self.lpm_dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.lpm_dim, bias=False)

        # Initialize weights
        default_linear_init(self.wq.weight)
        default_linear_init(self.wk.weight)
        default_linear_init(self.wv.weight)
        default_linear_init(self.wo.weight)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        keys, values = xk, xv

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def __call__(self, x):
        return self.forward(x)

class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 multiple_of: int,
                 ffn_dim_multiplier: float = None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # Initialize weights
        default_linear_init(self.w1.weight)
        default_linear_init(self.w2.weight)
        default_linear_init(self.w3.weight)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def __call__(self, x):
        return self.forward(x)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int,
                 lpm_dim: int,
                 n_heads: int,
                 multiple_of: int,
                 ffn_dim_multiplier: int = None):
        super().__init__()
        self.layer_id = layer_id
        self.lpm_dim = lpm_dim
        self.n_heads = n_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.attention = Attention(lpm_dim=self.lpm_dim, n_heads=self.n_heads)
        self.feed_forward = FeedForward(
            dim=self.lpm_dim,
            hidden_dim=4 * self.lpm_dim,
            multiple_of=self.multiple_of,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim=self.lpm_dim)
        self.ffn_norm = RMSNorm(dim=self.lpm_dim)

    def forward(
        self,
        x: torch.Tensor,
    ):

        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def __call__(self, x):
        return self.forward(x)

class LanguageProjectionModule(nn.Module):
    def __init__(self,
                 base_model_projection_out_features: int = 1280,
                 language_tokens: int = 10,
                 n_lpm_layers: int = 4,
                 lpm_dim: int = 768,
                 n_heads: int = 8,
                 multiple_of: int = 256,
    ):
        super().__init__()
        self.base_model_projection_out_features = base_model_projection_out_features
        self.language_tokens = language_tokens
        self.n_lpm_layers = n_lpm_layers
        self.lpm_dim= lpm_dim
        self.n_heads = n_heads
        self.multiple_of = multiple_of
        self.languages = ["kabyle", "arabic", "french", "wolof", "yoruba"]

        self.resample_tokens = nn.ParameterDict()
        self.encoder_proj1 = nn.ModuleDict()
        self.encoder_proj2 = nn.ModuleDict()
        #self.start_tag = nn.ParameterDict()
        #self.end_tag = nn.ParameterDict()

        self.universal_language_projection = nn.ModuleList()
        for layer_id in range(self.n_lpm_layers):
            self.universal_language_projection.append(
                TransformerBlock(layer_id=layer_id,
                                 lpm_dim=self.lpm_dim,
                                 n_heads=self.n_heads,
                                 multiple_of=self.multiple_of)
            )

        for language in self.languages:
            self.resample_tokens[language] = nn.Parameter(
                torch.empty([1, self.language_tokens, self.lpm_dim]))
            nn.init.normal_(self.resample_tokens[language], std=0.02)


            self.encoder_proj1[language] = nn.Sequential(
              nn.Linear(self.base_model_projection_out_features, self.lpm_dim),
              nn.LayerNorm(self.lpm_dim)
              )

            self.encoder_proj2[language] = nn.Sequential(
                nn.Linear(self.lpm_dim, self.base_model_projection_out_features),
                nn.LayerNorm(self.base_model_projection_out_features)
                )

            # self.start_tag[language] = nn.Parameter(torch.rand(1, 1, self.lpm_dim))
            # self.end_tag[language] = nn.Parameter(torch.rand(1, 1, self.lpm_dim))


    def forward(self, audio_features, language):
        _bsz, _, _ = audio_features.shape

        audio_feats = self.encoder_proj1[language](audio_features)
        audio_feats = torch.cat(
                [self.resample_tokens[language].repeat(_bsz, 1, 1), audio_feats], dim=1
                )


        for layer_id in range(self.n_lpm_layers):
              audio_feats = self.universal_language_projection[layer_id](audio_feats)


        audio_feats = audio_feats[:, self.resample_tokens[language].size(1):, :]  # Take only modal tokens
        audio_feats = self.encoder_proj2[language](audio_feats)


        return audio_feats

    def __call__(self, audio_features, language):
        return self.forward(audio_features, language)

def ForCTCLoss(
    logits, labels, input_lengths, label_lengths, blank_token_id: int, reduction: str = "mean"
):
    
    ctc_loss = nn.CTCLoss(blank=blank_token_id, reduction=reduction, zero_infinity=False)

    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (seq_len, batch, vocab_size)

    loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
    return loss

"""**<h2>2. UniWav</h2>**"""

class UniWav(nn.Module):
  def __init__(self,
                 base_model_name: str,
                 processor_name: str,
                 base_model_projection_out_features: int = 1280,
                 language_tokens: int = 10,
                 lpm_dim: int = 768,
                 n_lpm_layers: int = 4,
                 n_heads: int = 8,
                 multiple_of: int = 256,
                 dropout_rate: float = 0.1,
                 vocab_size: int = 204,
                 ignore_index: int = -100
                 ):

      """
      base_model_name: str, the name of the base model to use.
      processor_name: str, the name of the processor (tokenizer and feature extractor)
      base_model_projection_out_features: int, the output features of the base model projection layer.
      language_tokens: int, the number of language tokens to use.
      n_lpm_layers: int, the number of Language Projection Module layers to use.
      lpm_dim: int = 768, the dimension of the language projection module (hidden size)
      n_heads: int = 8, the number of multi-heads attention to use in the Language Projection Module layers.
      multiple_of: int = 256, the dimension of projection to use in the Language Projection Module layers.
      dropout_rate: float = 0.1, the rate of the dropout layer
      vocab_size: int = 204, the size of the vocabualary (used in lm_head layer)
      ignore_index: int = -100, the ignore index value for loss calculation.
      """

      super().__init__()

      self.base_model_name = base_model_name
      self.processor_name = processor_name
      self.base_model_projection_out_features = base_model_projection_out_features
      self.language_tokens = language_tokens
      self.n_lpm_layers = n_lpm_layers
      self.n_heads = n_heads
      self.multiple_of = multiple_of
      self.ignore_index = ignore_index
      self.dropout_rate = dropout_rate
      self.vocab_size = vocab_size
      self.lpm_dim = lpm_dim

      # Load the base model and processor
      self.base_model = Wav2Vec2ForCTC.from_pretrained(self.base_model_name)
      self.processor = AutoProcessor.from_pretrained(self.processor_name)


      # Language Projection Module
      self.language_projection_module = LanguageProjectionModule(
                 base_model_projection_out_features = self.base_model_projection_out_features,
                 language_tokens = self.language_tokens,
                 n_lpm_layers = self.n_lpm_layers,
                 lpm_dim = self.lpm_dim,
                 n_heads = self.n_heads,
                 multiple_of = self.multiple_of,
      )

      # Dropout
      self.dropout = nn.Dropout(self.dropout_rate)

      # LM Head
      self.lm_head = nn.Linear(in_features=self.base_model_projection_out_features, out_features=self.vocab_size, bias=True)


  def forward(self, audio, labels=None, language="wolof"):

      audio_features = self.base_model.wav2vec2.feature_extractor(audio["input_values"])
      audio_features = self.base_model.wav2vec2.feature_projection(audio_features.transpose(1, 2))

      audio_feats = self.language_projection_module(audio_features[0], language)

      attention_mask = self.base_model._get_feature_vector_attention_mask(audio_feats.shape[1],
                                         audio["attention_mask"],
                                          add_adapter=None)

      # Residual Connection
      audio_feats += audio_features[0]

      audio_features = self.base_model.wav2vec2.encoder(audio_feats, attention_mask)

      dropout = self.dropout(audio_features.last_hidden_state)

      logits = self.lm_head(dropout)

      outputs = {
          "logits": logits
      }
      if labels is not None:
          input_lengths = torch.tensor([logits.shape[1]] * logits.shape[0])
          label_lengths = torch.tensor([len(lbl[lbl != self.ignore_index]) for lbl in labels['input_ids']])

          loss = ForCTCLoss(
              logits, labels['input_ids'], input_lengths, label_lengths, blank_token_id=self.processor.tokenizer.pad_token_id
          )

          outputs["loss"] = loss

      return outputs


  def __call__(self, audio, labels, language):
      return self.forward(audio, labels, language)

"""**<h1>3. Data Load</h1>**"""

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 datasets,
                 processor,
                 sampling_rate=16_000,
                 *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.datasets = datasets
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __getitem__(self, index):
        text = self.datasets[index]["Text"]
        audio = self.datasets[index]["audio"]

        audio_processed = self.processor(audio["array"], sampling_rate=self.sampling_rate)
        text_tokenized = self.processor(text=text)

        item = {
            "input_values": audio_processed["input_values"][0],
            "input_ids": text_tokenized["input_ids"],
        }

        return item

    def __len__(self):
        return len(self.datasets)

processor = AutoProcessor.from_pretrained(CFG.processor_name)
def collate_fn(items):

    input_features = [{"input_values": feature["input_values"]} for feature in items]
    labels = [{"input_ids": label["input_ids"]} for label in items]

    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt"
    )

    labels = processor.pad(
        labels=labels,
        padding=True,
        return_tensors="pt"
    )

    labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

    batch["input_ids"] = labels

    return batch

def build_audio_loaders(datasets, processor, batch_size=4, num_workers=0, **kwargs):
    dataset = Dataset(datasets, processor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )

    return dataloader

"""**<h1>4. Train</h1>**"""

import re
chars_to_remove_regex = '[\,\.\;\:\"\“\%\‘\”\�\(\)\_\«\»]'

def remove_special_characters(batch):
    batch["Text"] = re.sub(chars_to_remove_regex, '', batch["Text"]).lower()
    return batch

def clean_text(batch):
    batch["Text"] = [re.sub(r'[\xa0\u202f\u200b]', ' ', text) for text in batch["Text"]]
    return batch


##Dataset wolof   
def load_wolof_dataset(subset=CFG.wolof_dataset_name):
    dataset = load_dataset(subset, trust_remote_code=True)
    dataset = dataset.rename_column("transcription", "Text")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    dataset = dataset.map(remove_special_characters)
    return dataset['train'], dataset['dev'], dataset['test']


##Dataset french

def load_french_dataset(subset=CFG.french_dataset_name, seed=42):
    dataset = load_dataset(subset, trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    dataset = dataset.map(remove_special_characters)
    dataset = dataset.map(clean_text, batched=True)
    test_split = dataset['test'].train_test_split(test_size=0.3, seed=seed)
    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': test_split['train'],
        'test': test_split['test'],
    })
    return dataset['train'], dataset['validation'], dataset['test']


##Dataset arabic

def load_arabic_dataset(subset=CFG.arabic_dataset_name, seed=42):
    dataset = load_dataset(subset, trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    dataset = dataset.map(remove_special_characters)
    test_split = dataset['test'].train_test_split(test_size=0.3, seed=seed)
    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': test_split['train'],
        'test': test_split['test'],
    })
    return dataset['train'], dataset['validation'], dataset['test']


##Dataset Kabyle

def load_kabyle_dataset(subset=CFG.kabyle_dataset_name, seed=42):
    dataset = load_dataset(subset, trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    dataset = dataset.map(remove_special_characters)
    dataset = dataset.map(clean_text, batched=True)
    test_split = dataset['test'].train_test_split(test_size=0.3, seed=seed)
    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': test_split['train'],
        'test': test_split['test'],
    })
    return dataset['train'], dataset['validation'], dataset['test']
    
    



def compute_metrics(wer_metric, bleu_metric, rouge_metric, pred_logits, labels, processor, ignore_index=-100):

    pred_ids = np.argmax(pred_logits.cpu().numpy(), axis=-1)

    labels[labels == ignore_index] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(labels, group_tokens=False)


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

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        # val: mean loss of the batch
        # count: Total of samples (sum of samples in all mini-batch)
        # sum: sum of loss (*count allow to pass to mean to sum)
        # avg: mean loss of all batches
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, gradient_accumulation_steps: int = 8, language="wolof"):
    loss_meter = AvgMeter()
    # List of dictionary, each dictionary is a batch
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    optimizer.zero_grad() 
    for idx, batch in enumerate(tqdm_object, 1):
        audio  = {k: v.to(CFG.device) for k, v in batch.items() if k != "input_ids"}
        labels = {k: v.to(CFG.device) for k, v in batch.items() if k == "input_ids"}

        outputs = model(audio, labels, language)
        loss = outputs["loss"]

        loss = loss / gradient_accumulation_steps  
        loss.backward()                           

        if idx % gradient_accumulation_steps == 0:
            optimizer.step()      
            optimizer.zero_grad() 
            if step == "batch":
                lr_scheduler.step()

        count = audio["input_values"].size(0)            
        loss_meter.update(
            loss.item() * gradient_accumulation_steps, 
            count
        )
        tqdm_object.set_postfix(
            train_loss=loss_meter.avg,
            lr=get_lr(optimizer)
        )

    return loss_meter


def val_epoch(model, val_loader, processor, wer_metric, bleu_metric, rouge_metric, ignore_index=-100, language="wolof"):
    loss_meter = AvgMeter(name="Loss")
    wer_meter = AvgMeter(name="WER")
    bleu_meter = AvgMeter(name="Bleu")
    rouge1_meter = AvgMeter(name="Rouge1")
    rouge2_meter = AvgMeter(name="Rouge2")
    rougeL_meter = AvgMeter(name="RougeL")

    tqdm_object = tqdm(val_loader, total=len(val_loader))

    for batch in tqdm_object:
        audio = {k: v.to(CFG.device) for k, v in batch.items() if k != "input_ids"}
        labels = {k: v.to(CFG.device) for k, v in batch.items() if k == "input_ids"}

        outputs = model(audio, labels, language)
        loss = outputs["loss"]

        metrics = compute_metrics(wer_metric, bleu_metric, rouge_metric, outputs["logits"], labels["input_ids"], processor, ignore_index=ignore_index)
        wer = metrics["wer"]
        bleu = metrics["bleu"]
        rouge1 = metrics["rouge"]["rouge1"]
        rouge2 = metrics["rouge"]["rouge2"]
        rougeL = metrics["rouge"]["rougeL"]

        count = audio["input_values"].size(0)
        loss_meter.update(loss.item(), count)
        wer_meter.update(wer, count)
        bleu_meter.update(bleu, count)
        rouge1_meter.update(rouge1, count)
        rouge2_meter.update(rouge2, count)
        rougeL_meter.update(rougeL, count)

        tqdm_object.set_postfix(val_loss=loss_meter.avg,
                                wer=wer_meter.avg,
                                bleu=bleu_meter.avg,
                                rouge1=rouge1_meter.avg,
                                rouge2=rouge2_meter.avg,
                                rougeL=rougeL_meter.avg)

    return loss_meter, wer_meter, bleu_meter, rouge1_meter, rouge2_meter, rougeL_meter


def test_epoch(model, test_loader, processor, wer_metric, bleu_metric, rouge_metric, ignore_index=-100, language="wolof"):
    loss_meter = AvgMeter(name="Test Loss")
    wer_meter = AvgMeter(name="Test WER")
    bleu_meter = AvgMeter(name="Test Bleu")
    rouge1_meter = AvgMeter(name="Test Rouge1")
    rouge2_meter = AvgMeter(name="Test Rouge2")
    rougeL_meter = AvgMeter(name="Test RougeL")

    tqdm_object = tqdm(test_loader, total=len(test_loader))

    for batch in tqdm_object:
        audio = {k: v.to(CFG.device) for k, v in batch.items() if k != "input_ids"}
        labels = {k: v.to(CFG.device) for k, v in batch.items() if k == "input_ids"}

        outputs = model(audio, labels, language)
        loss = outputs["loss"]

        metrics = compute_metrics(
            wer_metric, bleu_metric, rouge_metric,
            outputs["logits"], labels["input_ids"],
            processor, ignore_index
        )
        wer = metrics["wer"]
        bleu = metrics["bleu"]
        rouge1 = metrics["rouge"]["rouge1"]
        rouge2 = metrics["rouge"]["rouge2"]
        rougeL = metrics["rouge"]["rougeL"]

        count = audio["input_values"].size(0)
        loss_meter.update(loss.item(), count)
        wer_meter.update(wer, count)
        bleu_meter.update(bleu, count)
        rouge1_meter.update(rouge1, count)
        rouge2_meter.update(rouge2, count)
        rougeL_meter.update(rougeL, count)

        tqdm_object.set_postfix(
            test_loss=loss_meter.avg,
            wer=wer_meter.avg,
            bleu=bleu_meter.avg,
            rouge1=rouge1_meter.avg,
            rouge2=rouge2_meter.avg,
            rougeL=rougeL_meter.avg
        )

    return loss_meter, wer_meter, bleu_meter, rouge1_meter, rouge2_meter, rougeL_meter


def main():
    # Load processor
    processor = AutoProcessor.from_pretrained(CFG.processor_name)

    # Load datasets
    train_dataset_kabyle, val_dataset_kabyle, test_dataset_kabyle = load_kabyle_dataset()
    train_loader_kabyle =  build_audio_loaders(train_dataset_kabyle, processor)
    val_loader_kabyle = build_audio_loaders(val_dataset_kabyle, processor)
    test_loader_kabyle = build_audio_loaders(test_dataset_kabyle, processor)
    
    train_dataset_arabic, val_dataset_arabic, test_dataset_arabic = load_arabic_dataset()
    train_loader_arabic =  build_audio_loaders(train_dataset_arabic, processor)
    val_loader_arabic = build_audio_loaders(val_dataset_arabic, processor)
    test_loader_arabic = build_audio_loaders(test_dataset_arabic, processor)

    train_dataset_french, val_dataset_french, test_dataset_french = load_french_dataset()
    train_loader_french = build_audio_loaders(train_dataset_french, processor)
    val_loader_french = build_audio_loaders(val_dataset_french, processor)
    test_loader_french = build_audio_loaders(test_dataset_french, processor)

    train_dataset_wolof, val_dataset_wolof, test_dataset_wolof = load_wolof_dataset()
    train_loader_wolof = build_audio_loaders(train_dataset_wolof, processor)
    val_loader_wolof = build_audio_loaders(val_dataset_wolof, processor)
    test_loader_wolof = build_audio_loaders(test_dataset_wolof, processor)


    # Create model
    uniwav = UniWav(base_model_name=CFG.base_model_name, processor_name=CFG.processor_name)
    uniwav = uniwav.to(CFG.device)

    uniwav.load_state_dict(torch.load("../GPAfrica_unispeech_prog_french/best_kab.pt", map_location=CFG.device))

    # Freeze the base model
    for param in uniwav.base_model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False
    for param in uniwav.base_model.wav2vec2.feature_projection.parameters():
        param.requires_grad = False
    for param in uniwav.base_model.wav2vec2.encoder.parameters():
        param.requires_grad = False
    languages = ["yoruba"]
    for language in languages:
        for param in uniwav.language_projection_module.encoder_proj1[language].parameters():
            param.requires_grad = False
        for param in uniwav.language_projection_module.encoder_proj2[language].parameters():
            param.requires_grad = False
        uniwav.language_projection_module.resample_tokens[language].requires_grad = False

    adapter_weights = uniwav.base_model.wav2vec2._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True


    # Optimizer
    optimizer = torch.optim.AdamW(uniwav.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode="min", #patience=CFG.patience, 
      factor=CFG.factor
    )

    # Metrics
    wer_metric = load("wer")
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")


    # Train and validation
    #num_bad_epochs = 0
    
    #best_loss_kab = float('inf')
    #best_loss_ar = float('inf')
    #best_loss_french = float('inf')

    best_wer_kab = float('inf')
    best_wer_ar = float('inf')
    best_wer_french = float('inf')
    best_wer_wolof = float('inf')


    # Train the model
    kabyle_loader_cycle = cycle(train_loader_kabyle)
    arabic_loader_cycle = cycle(train_loader_arabic)
    french_loader_cycle = cycle(train_loader_french)

    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        uniwav.train()
        running_loss = 0.0
        optimizer.zero_grad()
        compt = 0
        for step, batch in enumerate(tqdm(train_loader_wolof, desc=f"wolof Epoch {epoch+1}")):
            audio = {k: v.to(CFG.device) for k, v in batch.items() if k != "input_ids"}
            labels = {k: v.to(CFG.device) for k, v in batch.items() if k == "input_ids"}

            outputs = uniwav(audio, labels, language="wolof")
            loss = outputs["loss"] / CFG.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                compt+=1
                #print("optimizer on wolof")
                optimizer.step()
                optimizer.zero_grad()
                if compt == 3:
                    for _ in range(8):
                        kabyle_batch = next(kabyle_loader_cycle)
                        audio_k = {k: v.to(CFG.device) for k, v in kabyle_batch.items() if k != "input_ids"}
                        labels_k = {k: v.to(CFG.device) for k, v in kabyle_batch.items() if k == "input_ids"}

                        outputs_k = uniwav(audio_k, labels_k, language="kabyle")
                        loss_k = outputs_k["loss"] / CFG.gradient_accumulation_steps
                        loss_k.backward()

                        if (_ + 1) % CFG.gradient_accumulation_steps == 0:
                            #print("optimizer on kabyle")
                            optimizer.step()
                            optimizer.zero_grad()

                    for _ in range(8):
                        arabic_batch = next(arabic_loader_cycle)
                        audio_k = {k: v.to(CFG.device) for k, v in arabic_batch.items() if k != "input_ids"}
                        labels_k = {k: v.to(CFG.device) for k, v in arabic_batch.items() if k == "input_ids"}

                        outputs_k = uniwav(audio_k, labels_k, language="arabic")
                        loss_k = outputs_k["loss"] / CFG.gradient_accumulation_steps
                        loss_k.backward()

                        if (_ + 1) % CFG.gradient_accumulation_steps == 0:
                            #print("optimizer on arabic")
                            optimizer.step()
                            optimizer.zero_grad()

                    for _ in range(8):
                        french_batch = next(french_loader_cycle)
                        audio_k = {k: v.to(CFG.device) for k, v in french_batch.items() if k != "input_ids"}
                        labels_k = {k: v.to(CFG.device) for k, v in french_batch.items() if k == "input_ids"}

                        outputs_k = uniwav(audio_k, labels_k, language="french")
                        loss_k = outputs_k["loss"] / CFG.gradient_accumulation_steps
                        loss_k.backward()

                        if (_ + 1) % CFG.gradient_accumulation_steps == 0:
                            #print("optimizer on french")
                            optimizer.step()
                            optimizer.zero_grad()
                            compt = 0

        avg_train_loss = running_loss / len(train_loader_wolof)
        print(f"Epoch: {epoch+1} train loss: {avg_train_loss:.4f}")

        uniwav.eval()

        with torch.no_grad():
            print("Validation on wolof...")
            val_loss_wolof, wer_wolof, bleu_wolof, rouge1_wolof, rouge2_wolof, rougeL_wolof = val_epoch(
                uniwav, val_loader_wolof, processor, wer_metric, bleu_metric, rouge_metric,
                ignore_index=CFG.ignore_index, language="wolof"
            )
            print(
                f"[wolof Validation Epoch {epoch+1}] "
                f"Loss: {val_loss_wolof.avg:.4f} | "
                f"WER: {wer_wolof.avg:.4f} | "
                f"BLEU: {bleu_wolof.avg:.4f} | "
                f"ROUGE-1: {rouge1_wolof.avg:.4f} | "
                f"ROUGE-2: {rouge2_wolof.avg:.4f} | "
                f"ROUGE-L: {rougeL_wolof.avg:.4f}"
            )

        with torch.no_grad():
            print("Validation on french...")
            val_loss_french, wer_french, bleu_french, rouge1_french, rouge2_french, rougeL_french = val_epoch(
                uniwav, val_loader_french, processor, wer_metric, bleu_metric, rouge_metric,
                ignore_index=CFG.ignore_index, language="french"
            )
            print(
                f"[french Validation Epoch {epoch+1}] "
                f"Loss: {val_loss_french.avg:.4f} | "
                f"WER: {wer_french.avg:.4f} | "
                f"BLEU: {bleu_french.avg:.4f} | "
                f"ROUGE-1: {rouge1_french.avg:.4f} | "
                f"ROUGE-2: {rouge2_french.avg:.4f} | "
                f"ROUGE-L: {rougeL_french.avg:.4f}"
            )


        with torch.no_grad():
            print("Validation on arabic...")
            val_loss_ar, wer_ar, bleu_ar, rouge1_ar, rouge2_ar, rougeL_ar = val_epoch(
                uniwav, val_loader_arabic, processor, wer_metric, bleu_metric, rouge_metric,
                ignore_index=CFG.ignore_index, language="arabic"
            )
            print(
                f"[arabic Validation Epoch {epoch+1}] "
                f"Loss: {val_loss_ar.avg:.4f} | "
                f"WER: {wer_ar.avg:.4f} | "
                f"BLEU: {bleu_ar.avg:.4f} | "
                f"ROUGE-1: {rouge1_ar.avg:.4f} | "
                f"ROUGE-2: {rouge2_ar.avg:.4f} | "
                f"ROUGE-L: {rougeL_ar.avg:.4f}"
            )

        with torch.no_grad():
            print("Validation on Kabyle...")
            val_loss_kab, wer_kab, bleu_kab, rouge1_kab, rouge2_kab, rougeL_kab = val_epoch(
                uniwav, val_loader_kabyle, processor, wer_metric, bleu_metric, rouge_metric,
                ignore_index=CFG.ignore_index, language="kabyle"
            )
            print(
                f"[Kabyle Validation Epoch {epoch+1}] "
                f"Loss: {val_loss_kab.avg:.4f} | "
                f"WER: {wer_kab.avg:.4f} | "
                f"BLEU: {bleu_kab.avg:.4f} | "
                f"ROUGE-1: {rouge1_kab.avg:.4f} | "
                f"ROUGE-2: {rouge2_kab.avg:.4f} | "
                f"ROUGE-L: {rougeL_kab.avg:.4f}"
            )

        if wer_wolof.avg < best_wer_wolof:
            best_wer_wolof = wer_wolof.avg
            torch.save(uniwav.state_dict(), "best_wolof.pt")
            print("Saved best wolof model!")

        if wer_french.avg < best_wer_french:
            best_wer_french = wer_french.avg
            torch.save(uniwav.state_dict(), "best_french.pt")
            print("Saved best french model!")

        if wer_ar.avg < best_wer_ar:
            best_wer_ar = wer_ar.avg
            torch.save(uniwav.state_dict(), "best_ar.pt")
            print("Saved best arabic model!")
        
        if wer_kab.avg < best_wer_kab:
            best_wer_kab = wer_kab.avg
            torch.save(uniwav.state_dict(), "best_kab.pt")
            print("Saved best kabyle model!")

        lr_scheduler.step(val_loss_wolof.avg)

    torch.save(uniwav.state_dict(), "last.pt")



    print("\n=== FINAL TEST ===")


    print("\nLoading best wolof model for final test...")
    uniwav.load_state_dict(torch.load("best_wolof.pt"))
    uniwav.eval()
    with torch.no_grad():
        print("Testing on wolof...")
        test_loss_wolof, wer_wolof, bleu_wolof, rouge1_wolof, rouge2_wolof, rougeL_wolof = val_epoch(
            uniwav, test_loader_wolof, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="wolof"
        )
        print(f"wolof Test -> Loss: {test_loss_wolof.avg:.4f}, WER: {wer_wolof.avg:.4f}, "
            f"BLEU: {bleu_wolof.avg:.4f}, ROUGE-1: {rouge1_wolof.avg:.4f}, "
            f"ROUGE-2: {rouge2_wolof.avg:.4f}, ROUGE-L: {rougeL_wolof.avg:.4f}")
        ################################################
        print("Testing on french...")
        test_loss_french, wer_french, bleu_french, rouge1_french, rouge2_french, rougeL_french = val_epoch(
            uniwav, test_loader_french, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="french"
        )
        print(f"french Test -> Loss: {test_loss_french.avg:.4f}, WER: {wer_french.avg:.4f}, "
            f"BLEU: {bleu_french.avg:.4f}, ROUGE-1: {rouge1_french.avg:.4f}, "
            f"ROUGE-2: {rouge2_french.avg:.4f}, ROUGE-L: {rougeL_french.avg:.4f}")
        ################################################
        print("Testing on arabic...")
        test_loss_ar, wer_ar, bleu_ar, rouge1_ar, rouge2_ar, rougeL_ar = val_epoch(
            uniwav, test_loader_arabic, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="arabic"
        )
        print(f"arabic Test -> Loss: {test_loss_ar.avg:.4f}, WER: {wer_ar.avg:.4f}, "
            f"BLEU: {bleu_ar.avg:.4f}, ROUGE-1: {rouge1_ar.avg:.4f}, "
            f"ROUGE-2: {rouge2_ar.avg:.4f}, ROUGE-L: {rougeL_ar.avg:.4f}")
        ################################################
        print("Testing on Kabyle...")
        test_loss_kab, wer_kab, bleu_kab, rouge1_kab, rouge2_kab, rougeL_kab = val_epoch(
            uniwav, test_loader_kabyle, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="kabyle"
        )
        print(f"Kabyle Test -> Loss: {test_loss_kab.avg:.4f}, WER: {wer_kab.avg:.4f}, "
            f"BLEU: {bleu_kab.avg:.4f}, ROUGE-1: {rouge1_kab.avg:.4f}, "
            f"ROUGE-2: {rouge2_kab.avg:.4f}, ROUGE-L: {rougeL_kab.avg:.4f}")


    print("\nLoading best french model for final test...")
    uniwav.load_state_dict(torch.load("best_french.pt"))
    uniwav.eval()
    with torch.no_grad():

        print("Testing on wolof...")
        test_loss_wolof, wer_wolof, bleu_wolof, rouge1_wolof, rouge2_wolof, rougeL_wolof = val_epoch(
            uniwav, test_loader_wolof, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="wolof"
        )
        print(f"wolof Test -> Loss: {test_loss_wolof.avg:.4f}, WER: {wer_wolof.avg:.4f}, "
            f"BLEU: {bleu_wolof.avg:.4f}, ROUGE-1: {rouge1_wolof.avg:.4f}, "
            f"ROUGE-2: {rouge2_wolof.avg:.4f}, ROUGE-L: {rougeL_wolof.avg:.4f}")
        ################################################
        print("Testing on french...")
        test_loss_french, wer_french, bleu_french, rouge1_french, rouge2_french, rougeL_french = val_epoch(
            uniwav, test_loader_french, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="french"
        )
        print(f"french Test -> Loss: {test_loss_french.avg:.4f}, WER: {wer_french.avg:.4f}, "
            f"BLEU: {bleu_french.avg:.4f}, ROUGE-1: {rouge1_french.avg:.4f}, "
            f"ROUGE-2: {rouge2_french.avg:.4f}, ROUGE-L: {rougeL_french.avg:.4f}")
        ################################################
        print("Testing on arabic...")
        test_loss_ar, wer_ar, bleu_ar, rouge1_ar, rouge2_ar, rougeL_ar = val_epoch(
            uniwav, test_loader_arabic, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="arabic"
        )
        print(f"arabic Test -> Loss: {test_loss_ar.avg:.4f}, WER: {wer_ar.avg:.4f}, "
            f"BLEU: {bleu_ar.avg:.4f}, ROUGE-1: {rouge1_ar.avg:.4f}, "
            f"ROUGE-2: {rouge2_ar.avg:.4f}, ROUGE-L: {rougeL_ar.avg:.4f}")
        ################################################
        print("Testing on Kabyle...")
        test_loss_kab, wer_kab, bleu_kab, rouge1_kab, rouge2_kab, rougeL_kab = val_epoch(
            uniwav, test_loader_kabyle, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="kabyle"
        )
        print(f"Kabyle Test -> Loss: {test_loss_kab.avg:.4f}, WER: {wer_kab.avg:.4f}, "
            f"BLEU: {bleu_kab.avg:.4f}, ROUGE-1: {rouge1_kab.avg:.4f}, "
            f"ROUGE-2: {rouge2_kab.avg:.4f}, ROUGE-L: {rougeL_kab.avg:.4f}")



    print("\nLoading best arabic model for final test...")
    uniwav.load_state_dict(torch.load("best_ar.pt"))
    uniwav.eval()
    with torch.no_grad():
        print("Testing on wolof...")
        test_loss_wolof, wer_wolof, bleu_wolof, rouge1_wolof, rouge2_wolof, rougeL_wolof = val_epoch(
            uniwav, test_loader_wolof, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="wolof"
        )
        print(f"wolof Test -> Loss: {test_loss_wolof.avg:.4f}, WER: {wer_wolof.avg:.4f}, "
            f"BLEU: {bleu_wolof.avg:.4f}, ROUGE-1: {rouge1_wolof.avg:.4f}, "
            f"ROUGE-2: {rouge2_wolof.avg:.4f}, ROUGE-L: {rougeL_wolof.avg:.4f}")
        ################################################
        print("Testing on french...")
        test_loss_french, wer_french, bleu_french, rouge1_french, rouge2_french, rougeL_french = val_epoch(
            uniwav, test_loader_french, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="french"
        )
        print(f"french Test -> Loss: {test_loss_french.avg:.4f}, WER: {wer_french.avg:.4f}, "
            f"BLEU: {bleu_french.avg:.4f}, ROUGE-1: {rouge1_french.avg:.4f}, "
            f"ROUGE-2: {rouge2_french.avg:.4f}, ROUGE-L: {rougeL_french.avg:.4f}")
        ################################################
        print("Testing on arabic...")
        test_loss_ar, wer_ar, bleu_ar, rouge1_ar, rouge2_ar, rougeL_ar = val_epoch(
            uniwav, test_loader_arabic, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="arabic"
        )
        print(f"arabic Test -> Loss: {test_loss_ar.avg:.4f}, WER: {wer_ar.avg:.4f}, "
            f"BLEU: {bleu_ar.avg:.4f}, ROUGE-1: {rouge1_ar.avg:.4f}, "
            f"ROUGE-2: {rouge2_ar.avg:.4f}, ROUGE-L: {rougeL_ar.avg:.4f}")
        ################################################
        print("Testing on Kabyle...")
        test_loss_kab, wer_kab, bleu_kab, rouge1_kab, rouge2_kab, rougeL_kab = val_epoch(
            uniwav, test_loader_kabyle, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="kabyle"
        )
        print(f"Kabyle Test -> Loss: {test_loss_kab.avg:.4f}, WER: {wer_kab.avg:.4f}, "
            f"BLEU: {bleu_kab.avg:.4f}, ROUGE-1: {rouge1_kab.avg:.4f}, "
            f"ROUGE-2: {rouge2_kab.avg:.4f}, ROUGE-L: {rougeL_kab.avg:.4f}")
        


    print("\nLoading best Kabyle model for final test...")
    uniwav.load_state_dict(torch.load("best_kab.pt"))
    uniwav.eval()
    with torch.no_grad():
        print("Testing on wolof...")
        test_loss_wolof, wer_wolof, bleu_wolof, rouge1_wolof, rouge2_wolof, rougeL_wolof = val_epoch(
            uniwav, test_loader_wolof, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="wolof"
        )
        print(f"wolof Test -> Loss: {test_loss_wolof.avg:.4f}, WER: {wer_wolof.avg:.4f}, "
            f"BLEU: {bleu_wolof.avg:.4f}, ROUGE-1: {rouge1_wolof.avg:.4f}, "
            f"ROUGE-2: {rouge2_wolof.avg:.4f}, ROUGE-L: {rougeL_wolof.avg:.4f}")
        ################################################
        print("Testing on french...")
        test_loss_french, wer_french, bleu_french, rouge1_french, rouge2_french, rougeL_french = val_epoch(
            uniwav, test_loader_french, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="french"
        )
        print(f"french Test -> Loss: {test_loss_french.avg:.4f}, WER: {wer_french.avg:.4f}, "
            f"BLEU: {bleu_french.avg:.4f}, ROUGE-1: {rouge1_french.avg:.4f}, "
            f"ROUGE-2: {rouge2_french.avg:.4f}, ROUGE-L: {rougeL_french.avg:.4f}")
        ################################################
        print("Testing on arabic...")
        test_loss_ar, wer_ar, bleu_ar, rouge1_ar, rouge2_ar, rougeL_ar = val_epoch(
            uniwav, test_loader_arabic, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="arabic"
        )
        print(f"arabic Test -> Loss: {test_loss_ar.avg:.4f}, WER: {wer_ar.avg:.4f}, "
            f"BLEU: {bleu_ar.avg:.4f}, ROUGE-1: {rouge1_ar.avg:.4f}, "
            f"ROUGE-2: {rouge2_ar.avg:.4f}, ROUGE-L: {rougeL_ar.avg:.4f}")
        ################################################
        print("Testing on Kabyle...")
        test_loss_kab, wer_kab, bleu_kab, rouge1_kab, rouge2_kab, rougeL_kab = val_epoch(
            uniwav, test_loader_kabyle, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="kabyle"
        )
        print(f"Kabyle Test -> Loss: {test_loss_kab.avg:.4f}, WER: {wer_kab.avg:.4f}, "
            f"BLEU: {bleu_kab.avg:.4f}, ROUGE-1: {rouge1_kab.avg:.4f}, "
            f"ROUGE-2: {rouge2_kab.avg:.4f}, ROUGE-L: {rougeL_kab.avg:.4f}")
        


    print("\nLoading last model for final test...")
    uniwav.load_state_dict(torch.load("last.pt"))
    uniwav.eval()
    with torch.no_grad():
        print("Testing on wolof...")
        test_loss_wolof, wer_wolof, bleu_wolof, rouge1_wolof, rouge2_wolof, rougeL_wolof = val_epoch(
            uniwav, test_loader_wolof, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="wolof"
        )
        print(f"wolof Test -> Loss: {test_loss_wolof.avg:.4f}, WER: {wer_wolof.avg:.4f}, "
            f"BLEU: {bleu_wolof.avg:.4f}, ROUGE-1: {rouge1_wolof.avg:.4f}, "
            f"ROUGE-2: {rouge2_wolof.avg:.4f}, ROUGE-L: {rougeL_wolof.avg:.4f}")
        ################################################
        print("Testing on french...")
        test_loss_french, wer_french, bleu_french, rouge1_french, rouge2_french, rougeL_french = val_epoch(
            uniwav, test_loader_french, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="french"
        )
        print(f"french Test -> Loss: {test_loss_french.avg:.4f}, WER: {wer_french.avg:.4f}, "
            f"BLEU: {bleu_french.avg:.4f}, ROUGE-1: {rouge1_french.avg:.4f}, "
            f"ROUGE-2: {rouge2_french.avg:.4f}, ROUGE-L: {rougeL_french.avg:.4f}")
        ################################################
        print("Testing on arabic...")
        test_loss_ar, wer_ar, bleu_ar, rouge1_ar, rouge2_ar, rougeL_ar = val_epoch(
            uniwav, test_loader_arabic, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="arabic"
        )
        print(f"arabic Test -> Loss: {test_loss_ar.avg:.4f}, WER: {wer_ar.avg:.4f}, "
            f"BLEU: {bleu_ar.avg:.4f}, ROUGE-1: {rouge1_ar.avg:.4f}, "
            f"ROUGE-2: {rouge2_ar.avg:.4f}, ROUGE-L: {rougeL_ar.avg:.4f}")
        ################################################
        print("Testing on Kabyle...")
        test_loss_kab, wer_kab, bleu_kab, rouge1_kab, rouge2_kab, rougeL_kab = val_epoch(
            uniwav, test_loader_kabyle, processor, wer_metric, bleu_metric, rouge_metric,
            ignore_index=CFG.ignore_index, language="kabyle"
        )
        print(f"Kabyle Test -> Loss: {test_loss_kab.avg:.4f}, WER: {wer_kab.avg:.4f}, "
            f"BLEU: {bleu_kab.avg:.4f}, ROUGE-1: {rouge1_kab.avg:.4f}, "
            f"ROUGE-2: {rouge2_kab.avg:.4f}, ROUGE-L: {rougeL_kab.avg:.4f}")


if __name__ == "__main__":
    main()