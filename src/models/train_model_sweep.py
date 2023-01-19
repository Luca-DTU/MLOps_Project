import datasets
import torch.cuda
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import omegaconf
#import hydra
import os
from src.data.make_dataset import yelp_dataset
from src.models.model import transformer
import wandb


# method
sweep_config = {
    'method': 'random'
}

# hyperparameters
parameters_dict = {
    'epochs': {
        'value': 1
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
}

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='huggingface_sweep')


# Define metric function
def compute_metrics(eval_pred):
    # Define metric
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
def load_training_cfg(cfg):
    info = cfg.model
    training_args = TrainingArguments(**info)
    return training_args

# def load_training_sweep(config):
#     training_args = TrainingArguments(
#         output_dir='vit-sweeps',
# 	    report_to='wandb',  # Turn on Weights & Biases logging
#         num_train_epochs=config.epochs,
#         learning_rate=config.learning_rate,
#         weight_decay=config.weight_decay,
#         save_strategy='steps',
#         evaluation_strategy='steps',
#         logging_strategy='steps'
#     )
#     return training_args

def main(config=None):
    with wandb.init(config=config):
        seed = 69
        train_size = 10
        test_size = 10
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        train_set = yelp_dataset(
            train=True,
            in_folder="data/raw",
            out_folder="data/processed",
            seed=seed,
            size=train_size,
        ).data

        test_set = yelp_dataset(
            train=False,
            in_folder="data/raw",
            out_folder="data/processed",
            seed=seed,
            size=test_size,
        ).data

        # Download the pretrained model
        model = transformer("models/pre_trained").to(device)
        # load training configuration from cfg file
        training_args = TrainingArguments(
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            save_strategy='steps',
            evaluation_strategy='steps',
            logging_strategy='steps',
            eval_accumulation_steps = 1,
            eval_steps = 1000
            )

        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=test_set,
            compute_metrics=compute_metrics,
        )

        # Train!
        trainer.train()
        #trainer.save_model("models/experiments")

wandb.agent(sweep_id, main, count=3)

# if __name__ == "__main__":
#     main()