from transformers import AutoModelForSequenceClassification
import os

def model():
    if len(os.listdir("models")) == 0:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
        model.save_pretrained("models")
    else:
        model = AutoModelForSequenceClassification.from_pretrained("src/models/pre_trained", num_labels=5)
    return model

model()
