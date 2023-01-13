from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

#Load data
dataset = load_dataset("yelp_review_full")
dataset["train"][100]

#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#Run tokenizer on dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

#Get a subset of dataset
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

#Download the pretrained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

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