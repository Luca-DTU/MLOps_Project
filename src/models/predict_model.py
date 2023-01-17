from src.models.model import transformer
from transformers import TrainingArguments, Trainer
import os
from src.data.make_dataset import yelp_dataset
from src.models.train_model import compute_metrics
import hydra
import numpy as np
from train_model import compute_metrics


@hydra.main(config_path=os.path.join(os.getcwd(), "conf"), config_name="config.yaml")
def predict_model(cfg):
    seed = cfg.data.seed
    size = cfg.predict.size
    input_filepath = cfg.data.input_filepath
    output_filepath = cfg.data.output_filepath
    path = cfg.predict.model_path

    test_args = TrainingArguments(
        output_dir="test_trainer",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=8,
        dataloader_drop_last=False
    )

    data = yelp_dataset(
        train=False,
        in_folder=input_filepath,
        out_folder=output_filepath,
        seed=seed,
        size=size,
    ).data

    model = transformer(path)
    trainer = Trainer(model,
                      args=test_args,
                      compute_metrics=compute_metrics)

    predictions, labels, metrics = trainer.predict(data)

    metrics["predict_samples"] = len(data)

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    predictions = np.argmax(predictions, axis=1)
    output_predict_file = os.path.join(os.getcwd(), cfg.predict.save_path, "predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                print(f"{index}\t{item}\t{labels[index]}")
                writer.write(f"{index}\t{item}\n")


predict_model()
