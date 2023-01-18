from transformers import AutoTokenizer, Trainer
from transformers import TrainingArguments, Trainer
import torch
from src.models.model import transformer
from src.data.make_dataset import yelp_dataset
import pdb


# Tokenizer - convert string to token that can be fed to the model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # Load tokenizer
def tokenize_function(examples):
   return tokenizer(examples, padding="max_length", truncation=True)


rev = "this food suck ass"

tok = tokenize_function(rev)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer("models/pre_trained").to(device)

from transformers import TextClassificationPipeline
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

pdb.set_trace()

test_args = TrainingArguments(
        output_dir="test_trainer",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=32,
        dataloader_drop_last=False
    )

trainer = Trainer(model,test_args)

data = yelp_dataset(
        train=False,
        in_folder="data/raw",
        out_folder="data/processed",
        seed=123,
        size=1,
    ).data

pdb.set_trace()

predictions, labels = trainer.predict(tok)

#print(tokenize_function(rev))