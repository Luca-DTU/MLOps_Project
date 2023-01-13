# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import datasets
from torch.utils.data import Dataset
import torch

class COCO(Dataset):
    def __init__(self, ingest, processed, test=False):
        self.ing = ingest
        self.proc = processed
        if test:
            self.subset = "train"
        else:
            self.subset = "new"
        try:
            self.data, self.labels = torch.load(
                os.path.join(processed, f"{self.subset}.pt")
            )
        except:
            self.data, self.labels = self.treat_data()
            torch.save(
                [self.data, self.labels],
                os.path.join(processed, f"{self.subset}.pt"),
            )

    def treat_data(self):
        files = os.listdir(f"data/raw/images/{self.subset}2017")
        for file in files:

        data = 0
        return data

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        return img.float(), target

