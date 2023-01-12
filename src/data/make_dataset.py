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


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    #TODO: This is just a copy of CLIP pre-run code.
    # Should be reformated to fit within a func
    COCO_DIR = os.path.join(os.getcwd(), "data")
    ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
