from transformers import AutoModelForSequenceClassification
import os

def model():
    if len(os.listdir("src/models/pre_trained")) == 0:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
        model.save_pretrained("src/models/pre_trained")
    else:
        model = AutoModelForSequenceClassification.from_pretrained("src/models/pre_trained", num_labels=5)
    return model

model()
