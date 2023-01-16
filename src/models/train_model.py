import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import omegaconf
import hydra
import os

from src.data.make_dataset import yelp_dataset
from src.models.model import transformer
cfg = omegaconf.OmegaConf.load('conf/config.yaml')

#Define metric function
def compute_metrics(eval_pred):
    #Define metric
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#Define training arguments
@hydra.main(config_path = os.path.join(os.getcwd(),'conf'), config_name='config.yaml')
def load_training_cfg(cfg):
    info = cfg.model
    training_args = TrainingArguments(None,**info)
    return training_args

def main():
    seed = cfg.data.seed
    size = cfg.data.size
    train_set = yelp_dataset(train=True, in_folder=cfg.data.input_filepath, out_folder=cfg.data.output_filepath,seed = seed,size = size).data
    test_set = yelp_dataset(train=False, in_folder=cfg.data.input_filepath, out_folder=cfg.data.output_filepath,seed = seed,size = size).data

    # Download the pretrained model
    model = transformer("models/pre_trained")
    # load training configuration from cfg file
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
    trainer.save_model("models/experiments")

if __name__=='__main__':
    main()