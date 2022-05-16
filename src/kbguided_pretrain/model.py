import os
import sys
import numpy as np
import tqdm
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from transformers import BartTokenizer, BartForConditionalGeneration
from dataclasses import dataclass

@dataclass
class modifiedLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        num_active_elements = padding_mask.numel() - padding_mask.long().sum()

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class BioBARTPTModel:

    def __init__(self, args):
        
        bart_config = AutoConfig.from_pretrained(args.config['bart_model_file'])
        
        self.label_smoother = modifiedLabelSmoother(args.label_smoothing_factor)
        self.config = args.config
        self.network = BartForConditionalGeneration.from_pretrained(args.config['bart_model_file'])
        self.tokenizer = BartTokenizer.from_pretrained(args.config['bart_token_file'])
        # self.tokenizer = Tokenizer.from_file(args.config['bart_model_file'] + '/tokenizer.json')

    def set_device(self, device):
        self.device = device

    def save(self, filename: str):
        os.makedirs(filename, exist_ok=True)
        self.network.module.save_pretrained(filename)
        self.tokenizer.save_pretrained(filename)
        return 

    def load(self, model_state_dict: str):
        return self.network.module.load_state_dict(
            torch.load(model_state_dict,
                       map_location=lambda storage, loc: storage))

    def move_batch(self, batch, non_blocking=False):
        return batch.to(self.device, non_blocking)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save_bert(self, filename: str):
        return torch.save(self.bert_encoder.state_dict(), filename)

    def to(self, device):
        assert isinstance(device, torch.device)
        self.network.to(device)

    def half(self):
        self.network.half()