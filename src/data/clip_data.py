import os
import datasets

COCO_DIR = os.path.join(os.getcwd(), "data", "zips")
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)
print(ds)