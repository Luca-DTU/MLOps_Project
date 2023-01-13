import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from src.data.make_dataset import yelp_dataset
from src.models.model import model


train_set = yelp_dataset(train=True, in_folder="data/raw", out_folder="data/processed")
test_set = yelp_dataset(train=False, in_folder="data/raw", out_folder="data/processed")

#Download the pretrained model
model = model()

#Define metric
metric = evaluate.load("accuracy")

#Define metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#Define training arguments
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

#Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

#Train!
trainer.train()