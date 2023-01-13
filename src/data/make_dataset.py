# from datasets import load_dataset
# dataset = load_dataset("yelp_review_full")
# dataset.save_to_disk('data/raw')
# #to load use
# #datasets.load_from_disk('data/raw')


# -*- coding: utf-8 -*-
import logging
import click
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import hydra
import os


class yelp_dataset(Dataset):
    def __init__(cfg,self, train: bool, in_folder: str = "", out_folder: str = "",seed : int = 0 ,size : int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.size = size
        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder

        if self.out_folder:  # try loading from proprocessed
            try:
                self.load_preprocessed()
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                pass

        self.download_data()

        ########## Preprocessing ##########
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") #Load tokenizer
        #Run tokenizer on dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        self.tokenized_datasets = self.dataset.map(tokenize_function, batched=True)
        #Get a subset of dataset
        split = "train" if self.train else "test"
        self.data = self.tokenized_datasets[split].shuffle(seed=self.seed).select(range(self.size))
        ##################################

        if self.out_folder:
            self.save_preprocessed()

    def save_preprocessed(self) -> None:
        split = "/train" if self.train else "/test"
        self.data.save_to_disk(self.out_folder+split) #Save to folder

    def load_preprocessed(self) -> None:
        split = "/train" if self.train else "/test"
        try:
            self.data = datasets.load_from_disk(self.out_folder+split)
        except:
            raise ValueError("No preprocessed files found")

    def download_data(self) -> None:
        try:
            self.dataset = datasets.load_from_disk(self.in_folder) #Load already downloaded data
        except:
            self.dataset = load_dataset("yelp_review_full") #Download data
            self.dataset.save_to_disk(self.in_folder) #Save to folder





@hydra.main(config_path = os.path.join(os.getcwd(),'conf'), config_name='config.yaml')
def main(cfg) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    seed = cfg.data.seed
    size = cfg.data.size
    input_filepath = cfg.data.input_filepath
    output_filepath = cfg.data.output_filepath

    train = yelp_dataset(train=True, in_folder=input_filepath, out_folder=output_filepath,seed = seed,size=size)

    test = yelp_dataset(train=False, in_folder=input_filepath, out_folder=output_filepath,seed=seed,size=size)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()