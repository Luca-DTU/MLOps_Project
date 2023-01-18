from fastapi import FastAPI
from src.models.model import transformer
from http import HTTPStatus
import omegaconf
import numpy as np
import os
import json
from transformers import TextClassificationPipeline
from transformers import AutoTokenizer
import torch

app = FastAPI()
cfg = omegaconf.OmegaConf.load("conf/config.yaml")
len_max = cfg.predict.len_max
path = cfg.predict.model_path
history_path = cfg.predict.save_path


@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def predict(model,string):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")  # Load tokenizer
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    out = pipe(string)
    return out

def store_predictions(string,output):
    output_predict_folder = os.path.join(os.getcwd(), cfg.predict.save_path)
    output_predict_file = os.path.join(output_predict_folder, "deployed_predictions.json")
    if not os.path.exists(output_predict_folder):
        os.makedirs(output_predict_folder)
    dict_obj = {string : output}
    try:
        with open(output_predict_file, "r") as outfile:
            data = json.load(outfile)
            data.update(dict_obj)
        with open(output_predict_file,"w") as outfile:
            json.dump(data,outfile)
    except json.decoder.JSONDecodeError as e:
        with open(output_predict_file, "w") as outfile:
            json.dump(dict_obj, outfile)
    except FileNotFoundError:
        os.mknod(output_predict_file)
        with open(output_predict_file, "w") as outfile:
            json.dump(dict_obj, outfile)


@app.get("/input")
def read_string(string: str):
    # check if string is within a number of characters
    length = len(string)
    if length > len_max:
        return f"The input is too long, please pass a smaller input, the character limit is {len_max}, got {length}"
    # standard API process
    model = transformer(path)
    output = predict(model,string)
    store_predictions(string,output)
    return output



