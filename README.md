MLOps_Project
==============================

Project for the Machine Learning Operations course at DTU

### Members
Luca Bertolani s220647 \
Jens Christian Bang Gribsvad s174477 \
Benjamin Jenfort Henriksen s214129 \
Poul Gunnar Pii Svane s153529 \
Andreas With Aspe s174107

### Overall goal of the project
The goal of the project is to generate image captions by fine tuning the CLIP model.
### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
We are going to use the Transformers framework
### How to you intend to include the framework into your project
The CLIP model is pre-trained and formulated by Transformers. We will initially finetune the model by further training the pre-trained model on the cocodataset to finetune it to the pictures and annotations. If possible we would also like to fully train the clip network for our given dataset, but only if there is enough time as our main focus is the operations. 

### What data are you going to run on (initially, may change)
We are going to run on the coco dataset, which is a comprehensive image dataset: https://cocodataset.org/
### What deep learning models do you expect to use
We are going to use the CLIP model: https://github.com/openai/CLIP
## Project checklist

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed very
point on the checklist for the exam.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github
