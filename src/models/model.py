from transformers import AutoModelForSequenceClassification
import os

def Transformer():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("models/pre_trained", num_labels=5)
    except:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
        model.save_pretrained("models/pre_trained")
    return model
