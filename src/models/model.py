from transformers import AutoModelForSequenceClassification


def transformer(path=None, num_labels=5):
    """
    Current file location is models/pre_trained
    """
    if not path:
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=num_labels
        )
        model.save_pretrained(path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels
        )
    return model
