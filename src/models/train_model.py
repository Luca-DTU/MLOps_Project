import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

import hydra
import os

from src.data.make_dataset import yelp_dataset
from src.models.model import Transformer



train_set = yelp_dataset(train=True, in_folder="data/raw", out_folder="data/processed")
test_set = yelp_dataset(train=False, in_folder="data/raw", out_folder="data/processed")

#Download the pretrained model
model = Transformer()

#Define metric
metric = evaluate.load("accuracy")

#Define metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#Define training arguments
@hydra.main(config_path = os.path.join(os.getcwd(),'conf'), config_name='config.yaml')
def load_training_cfg(cfg):
    info = cfg.model
    training_args = TrainingArguments(**info)
    return training_args

training_args = load_training_cfg()

#Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics,
)

#Train!
trainer.train()