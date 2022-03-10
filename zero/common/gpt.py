import copy
import os
import random
from functools import partial
from itertools import chain

import numpy as np
import torch
from datasets import load_dataset, set_progress_bar_enabled
from torch.distributed import get_world_size
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, default_data_collator)
from transformers.optimization import get_cosine_schedule_with_warmup

DATA_PATH = os.environ['DATA']
TOKEN_PATH = os.environ['TOKENIZER']
TOKEN_MODE = 'concat'
BATCH_SIZE = 4
SEQ_LEN = 1024
VOCAB_SIZE = 50257
HIDDEN_SIZE = 4096
NUM_HEADS = 16
DEPTH = 50
LEARNING_RATE = 0.00015
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 2
WARMUP_EPOCHS = 1


def build_data():
    set_progress_bar_enabled(False)
    dataset = load_dataset('wikitext', 'wikitext-2-v1', cache_dir='./tmp')
    tokenizer = GPT2Tokenizer(vocab_file=TOKEN_PATH + '/vocab.json', merges_file=TOKEN_PATH + '/merges.txt')

    def tokenize(examples, mode='concat'):
        assert mode in ['concat', 'pad']

        if mode == 'concat':
            examples = tokenizer(examples['text'])
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= SEQ_LEN:
                total_length = (total_length // SEQ_LEN) * SEQ_LEN

            result = {
                k: [t[i:i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
                for k, t in concatenated_examples.items()
            }
        else:
            tokenizer.pad_token = tokenizer.unk_token
            result = tokenizer(examples, padding=True, truncation=True, max_length=SEQ_LEN, return_tensors='pt')

        result["labels"] = copy.deepcopy(result["input_ids"])

        return result

    tokenized_dataset = dataset.map(partial(tokenize, mode=TOKEN_MODE), batched=True, num_proc=16, load_from_cache_file=False, remove_columns='text')

    global VOCAB_SIZE
    VOCAB_SIZE = len(tokenizer)

    world_size = get_world_size()

    def seed_worker(_):
        worker_seed = 1024
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    train_sampler = DistributedSampler(tokenized_dataset['train'], shuffle=True) if world_size > 1 else None
    train_data = DataLoader(tokenized_dataset['train'],
                            shuffle=(train_sampler is None),
                            sampler=train_sampler,
                            drop_last=True,
                            collate_fn=default_data_collator,
                            worker_init_fn=seed_worker,
                            batch_size=BATCH_SIZE,
                            num_workers=4,
                            pin_memory=True)
    test_sampler = DistributedSampler(tokenized_dataset['validation'], shuffle=False) if world_size > 1 else None
    test_data = DataLoader(tokenized_dataset['validation'],
                           sampler=test_sampler,
                           collate_fn=default_data_collator,
                           worker_init_fn=seed_worker,
                           batch_size=BATCH_SIZE,
                           num_workers=4,
                           pin_memory=True)

    return train_data, test_data


class ModelFromHF(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.module = GPT2LMHeadModel(config)
        self.module.transformer.gradient_checkpointing = True

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return output.logits


def build_model():
    config = GPT2Config(vocab_size=VOCAB_SIZE, n_positions=SEQ_LEN, n_embd=HIDDEN_SIZE, n_layer=DEPTH, n_head=NUM_HEADS)

    model = ModelFromHF(config)

    return model


class GPTLMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def build_loss():
    return GPTLMLoss()


def build_optimizer(params):
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return optimizer


def build_scheduler(epoch_steps, optimizer):
    max_steps = epoch_steps * NUM_EPOCHS
    warmup_steps = epoch_steps * WARMUP_EPOCHS
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warmup_steps,
                                                   num_training_steps=max_steps)

    return lr_scheduler


def gpt_builder():
    return build_data, build_model, build_loss, build_optimizer, build_scheduler
