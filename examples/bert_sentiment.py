import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from sklearn import metrics, model_selection
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


class CFG:
    DEVICE = "cuda"
    MAX_LEN = 64
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    N_SPLITS = 3
    EPOCHS = 3
    BERT_PATH = "/work/models/bert-base-japanese"
    MODEL_PATH = "/work/output/model.bin"
    TRAINING_FILE = "/work/input/train.csv"
    TEST_FILE = "/work/input/test.csv"
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(BERT_PATH)


class Dataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = CFG.TOKENIZER
        self.max_len = CFG.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=CFG.MAX_LEN,
            padding='max_length',
            return_tensors='pt'
        )

        ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]

        return {
            "ids": ids,
            "mask": mask,
            "targets": torch.tensor(self.target[index], dtype=torch.float),
        }


class BertBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(CFG.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask):
        bert_output = self.bert(input_ids=ids, attention_mask=mask)
        x = bert_output.last_hidden_state[:, 0, :]
        bo = self.bert_drop(x)
        output = self.out(bo)
        return output


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for _, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            outputs = torch.squeeze(outputs)
            fin_targets.extend(targets.cpu().detach().numpy().copy())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().copy())
    return fin_outputs, fin_targets
