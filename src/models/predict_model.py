from src.models.model import transformer
from transformers import TrainingArguments, Trainer


def predict_model(path=None):
    model = transformer(path)
    test_dataset = 0
    trainer = Trainer(model)
