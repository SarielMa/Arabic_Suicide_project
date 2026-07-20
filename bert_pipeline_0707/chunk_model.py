"""Chunked long-document sequence classification for BERT encoders.

A transcript longer than 512 tokens is split into several ~512-token windows.
Every window is encoded by the same Arabic BERT; the per-window [CLS] vectors
are mean-pooled across windows and passed to a classification head. This lets a
512-cap encoder read the whole call while keeping the pretrained Arabic weights.

Contains:
  * ChunkedModelForSequenceClassification — a PreTrainedModel (so save/load and
    the HF Trainer work normally).
  * build_chunks / ChunkDataset / ChunkCollator — data plumbing.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


def build_chunks(text, tokenizer, max_length, max_chunks, truncation):
    """Tokenize `text` and split into <=max_chunks windows with special tokens.

    truncation="head" keeps the first windows, "tail" keeps the last ones.
    Returns a list of input-id lists (one per window).
    """
    reserve = tokenizer.num_special_tokens_to_add(pair=False)  # e.g. [CLS]+[SEP]
    window = max_length - reserve
    body = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )["input_ids"]

    windows = [body[i : i + window] for i in range(0, max(len(body), 1), window)]
    if len(windows) > max_chunks:
        windows = windows[:max_chunks] if truncation == "head" else windows[-max_chunks:]

    return [tokenizer.build_inputs_with_special_tokens(w) for w in windows]


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunked_inputs, labels):
        self.chunked_inputs = chunked_inputs  # list[list[list[int]]]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"chunks": self.chunked_inputs[idx], "label": self.labels[idx]}


class ChunkCollator:
    """Pads a batch to (B, C, L) plus a (B, C) mask of real vs. padding chunks."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, batch):
        max_c = max(len(b["chunks"]) for b in batch)
        max_l = max((len(c) for b in batch for c in b["chunks"]), default=1)
        input_ids, attn, chunk_mask, labels = [], [], [], []
        for b in batch:
            doc_ids, doc_attn, doc_cm = [], [], []
            for chunk in b["chunks"]:
                pad = max_l - len(chunk)
                doc_ids.append(chunk + [self.pad_id] * pad)
                doc_attn.append([1] * len(chunk) + [0] * pad)
                doc_cm.append(1)
            for _ in range(max_c - len(b["chunks"])):  # pad missing chunks
                doc_ids.append([self.pad_id] * max_l)
                doc_attn.append([0] * max_l)
                doc_cm.append(0)
            input_ids.append(doc_ids)
            attn.append(doc_attn)
            chunk_mask.append(doc_cm)
            labels.append(b["label"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "chunk_mask": torch.tensor(chunk_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ChunkedModelForSequenceClassification(PreTrainedModel):
    """Encoder + mean-pool over chunk [CLS] vectors + linear classifier."""

    # AutoConfig rebuilds the correct encoder config (bert/roberta/...) on load.
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = AutoModel.from_config(config)
        dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.class_weights = None  # set externally for imbalanced loss
        self.post_init()

    @classmethod
    def from_encoder(cls, model_name, num_labels=2):
        """Build with a pretrained encoder and a fresh classification head."""
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = cls(config)
        model.encoder = AutoModel.from_pretrained(model_name)
        return model

    def forward(self, input_ids, attention_mask, chunk_mask=None, labels=None):
        b, c, length = input_ids.shape
        flat_ids = input_ids.view(b * c, length)
        flat_attn = attention_mask.view(b * c, length)
        out = self.encoder(input_ids=flat_ids, attention_mask=flat_attn)
        cls = out.last_hidden_state[:, 0]  # [CLS] per chunk -> (B*C, H)
        cls = cls.view(b, c, -1)

        if chunk_mask is None:
            chunk_mask = torch.ones(b, c, device=cls.device)
        m = chunk_mask.unsqueeze(-1).to(cls.dtype)  # (B, C, 1)
        pooled = (cls * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # mean over real chunks

        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            weight = (
                self.class_weights.to(logits.device)
                if self.class_weights is not None else None
            )
            loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return SequenceClassifierOutput(loss=loss, logits=logits)
