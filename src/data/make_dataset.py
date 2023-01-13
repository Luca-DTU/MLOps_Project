import datasets
from datasets import load_dataset
dataset = load_dataset("yelp_review_full")
dataset.save_to_disk('data/raw')
#to load use
#datasets.load_from_disk('data/raw')