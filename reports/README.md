---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 28 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s220647, s174477, s214129, s153529, s174107 ---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We have used the Hugginface Transformers framework which reduced the amount of boilerplate in the codebase and offered a lot of ad hoc integration with external software. For example it logged experiments to Weights & Biases by default. Another example of a very noticeable reduction in boilerplate is in the inference part of the project, it was useful when we created the fastApi since a wrapper named 'TextClassificationPipeline' could take in a tokenizer and a model and then predict based on a string-input directly in one line of code. This way we avoided to do complicated data set-up. Moreover we used the trainer class to train the model. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

--- We have created a requirements.txt file which keeps track of the package dependencies that is required to run the project. So for a new team member to get started with the project the following command must be run: 'pip install -r requirements.txt'. We have created the requirements file by autogeneration from our conda environment using the command 'conda list -e > requirements.txt'. This command scans our conda environment and fills out the requirements file with the packages one need to create the same environment. Furthermore, based on the cookiecutter structure we created a second requirements file which is used specifically for the additional requirements to run tests.---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- We initialized the project with the cookiecutter structure. We have filled out the src/data/ folder by completing the 'make_dataset.py' file and also filled out the src/models folder by completing the 'model.py', 'predict_model.py', 'train_model.py' and 'train_model_sweep.py' files. We have also added new folders like '.github', '.dvc' and 'conf'. .github takes care of unittesting and workflows, '.dvc' specifies the setup for the dvc and push/pulling data and lastly the 'conf' folder takes care of the configuration file for running the project. 
We have a few dockerfiles, yaml files for the Vertex AI runs a main.py inference file in the root directory, and we modified the makefile as well. We do have some empty folders which we could delete but we have kept them in case we suddenly would use them. ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- We have used flake8, isort and black to ensure a format that complies with pep8 standards and that all of our code look similar. We have implemented these tools in the workflow files such that when we push/pull the code will automatically first run black and then test for pep8 using flake8 in one workflow and runs isort in another workflow. These concepts matter since people code in different ways and if you work with a lot of people it is important you can understand others code. Tools like flake8 and black helps you understand other peoples code. ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer:

--- We have created 9 test unit test for this project, 4 tests that check the data and 5 tests that check the model. The data tests we have implemented check if the data folder exists, if the preprocessed data exists, if the shape of the data is correct and check and if the data consists of at least 26 characters. For the model tests we check whether the device is cuda or CPU, have labels, the size of the model and the model embeddings. 
We have implemented the tests in github workflows. ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- The total code coverage is 66% as reported by running the coverage report, which is far from 100%.  
If we had a 100% coverage we would have no reason to believe that our code would be error free, the reason for this are multiple.
The tests can cover all of the code but at the same time not pass, even if everything passed, the tests could just be "bad" tests and not really cover every aspect or every relation between the objects. Moreover the data can be corrupted and the tests on the code would not be able to catch it. ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- We exploited the potential of git extensively by using branches and pull requests for our personal work and when working together on larger subprojects.
An example of how we did that is when towards the end of the project we separated in two subgroups and consequently in two branches to work on the deployment part and on setting up the training on google cloud computing platform.
We have been consistent in committing frequently to minimize the merge conflicts. 
What we haven't done however and realized later it would have been a good approach is to use branches in the data versioning control, where we only worked in the master branch. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- As mentioned above we have used data versioning control on our data and on our model, which was very beneficial for sharing the correct version of the data and making sure that we all had access to the same subset and version of both the raw and the processed data. Similarly we stored our models in dvc which allowed us to store the pretrained model and to make sure that we were all using the same fine-tuned model (i.e. from the same training run).
We were however not fully using the potentialities of dvc since in the creation of docker images we just used the versions defined at the time instead of setting up the log in to gcp and the pull of the data. Moreover one member never actually managed to get access to our remote bucket using dvc and had to store the data locally. ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- We have constructed 3 workflow files for flake8, isort, and tests. We have thus integrated a continuous integration setup where we test the code for structure, we test some of the essential scripts for producing expected outputs, and isort to sort the imports. We do test multiple operating systems, namely the latest versions of macos, ubuntu and windows. We also cached our packages and run our workflows on four different Python versions (3.8 and 3.9). You can check out our workflows https://github.com/Luca-DTU/MLOps_Project/actions. We have not managed to solve the integration of the data handling through dvc in the workflows to be able to succesfully run the unit tests in time for the presentation so right now the unit tests workflow just fail everytime. ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- We made a config.yaml file and integrated this with hydra. However, we decided to primarily use wandb for logging so we removed the output-folder generated by hydra from our repository. We did not make any argparser but we changed all our parameters for each run in the config.yaml file. We also implemented hyperparameter sweeping from wandb. Here we made an in-script configuration where we examines a random combination of learning rate and weight decay. We modified the make-file, so you can run 'make sweep'. ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- Yes we have used a config file which states the hyperparameters such that we know the exact values when running an experiment. To reproduce experiments the user will then have to define hyperparameters by calling the config file.  Contrary, when we did a hyperparameter fine tuning we used WANDB which logs the exact values for each search. Thus the user can reproduce the experiments based on these logs. Moreover, when we created a fastApi we utilized docker by first building an image which then can be build into a container. Having this container also ensures reproducibility since it doesn't change among different users. ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

At the first picture below we can see a screenshot of a hyperparameter sweep. We can call this with the command 'make sweep'. We only did it on a very small subset of our data and screened for very few parameters, since it takes a long time to do hyperparameter sweeping. We chose a subset of the data consisting of only 100 observations for the training and test set respectively. We searched for optimal parameters with the method 'random', which equal probability of the weight-decay being 0.1, 0.2, 0.3, 0.4 and 0.5 and the learning rate to be in the interval [1e-5, 1e-3] with a uniform probability distribution. We ran 2 epochs and made 10 trials. For this experiment we get that the best model has the parameters learning-rate = $3.194\cdot 10^{-5}$ and weight_decay = 0.1. This gives us an accuracy of 0.23. This is pretty bad, because we only have 5 different labels so it basically means that the model predictions are random. This makes sense since we are training on so little data and only running 2 epochs. But we could just choose to train on the full data with more epochs, and the hyperparameter sweep we've implemented will work and can be a useful way of tuning the model.

![Alt text](figures/sweep.png?raw=true "sweep")

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- For our project we used docker a couple of times. We created a docker image and build a container for traning locally on a GPU. Moreover we also used docker in the deployment phase of the project where we created a fastApi. We build an image of the app-script named 'main.py' by using the command 'docker build -f triggerDeployment.dockerfile . -t fastapi-test:latest' and then we pushed this image to cloud by using the command 'docker tag fastapi-test gcr.io/<project-id>/fastapi-test' and 'docker push gcr.io/<project-id>/fastapi-test'. Then we can use the google cloud run to run the image which is now found in the google cloud container registry. A link to the docker file: gcr.io/mlops-project-374413/fastapi-test. --- 

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- In this group we are using VS code and Pycharm and we have used the built-in debugger for both IDE's. We also configured a launch.json in VS code when needed in order to debug a code that required arguments from the command line. We have also used set_trace with pdb. We naturally also did debugging with print statements, especially in the early stage development of the code. We also used the internet a lot while running into bugs and here we often got a clue about what could be wrong, which we could use to either solve it straightaway or narrow down the problem to set breakpoints at more important places in our code. We did no do any profiling, since most of our programme is actually handled by the Huggingface framework. Therefore we expect their code to be already optimised and our own code is not very complex. ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- We have used cloud build, cloud bucket, container registry, cloud vertex ai. Cloud build is a service where we can set up a trigger connected to a github repo that can automatically build a container each time a person pushes to the repo. The bucket is a service where one can store data. The registry is a service where you can find your containers. Vertex ai is a deployment interface where you can run containers within the cloud- it creates a virtual machine and runs a container. Furthermore we also used cloud run to deploy our fastAPI for using the trained model for perdicting ratings for food reviews. ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine through the vertex ai with a pytorch setup. We trained our virtual machine through vertex ai with CPU however we stopped it since it took too long. We also implemented to run with GPU but we encountered a domain issue which will take some time to fix.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:


![Alt text](figures/gcp_bucket.png?raw=true "gcp_bucket")

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Alt text](figures/gcp_registry.PNG?raw=true "gcp_registry")

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Alt text](figures/cloudbuild_history.png?raw=true "cloudbuild_history")

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- We managed to deploy the model both locally and in the cloud. First we wanted to make sure that the deployment worked locally before we pushed it to the cloud. When we got it to work locally we pushed to the cloud and it worked there. The deployment are based on a docker image of our fastApi. That image is then stored in google cloud container registry and the google cloud run invokes the image. Entering this link https://fastapi-test-nprrq6fuhq-ew.a.run.app/ you can add '/docs' to the url which will send you to fastApi interface. Here you can send request to the model and it will return either an error or a succcesful prediction. ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- We did not get so far as to implement any monitoring. Our project involves the prediction of the number of stars in a restaurant review given the review text.
There is not so much that actually changes over time for the kind of problem that we deal with. There could be some minor data drifting related to generational slang and overall how a language evolves in time. This however is far beyond the time scope that this project might take and we don't think it is too relevant.
In any case we do store the historical predictions given by the API in a json file within the deployed container, so that if someone would need to analyse that, there would be the option.  ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- The bucket ended up costing 0.47 Euro, the Cloud run to host the fastapi is costing 0.01 Euro and the network cost for s220647 were 0.02 Euro .
The user that set up the training runs is running into some IT problems right now and cannot see the cost, but he remembers it was around 2 Euros.---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![Alt text](figures/overview.png?raw=true "overview")

--- We start off looking at our local computer. From this, we initialized the project with GitHub and set up GCP and DVC. When running our code in practice we have workflow files that test our code for various properties whenever we push or pull from GitHub. The dataset is downloaded from DVC from a data bucket on our google cloud. GCP also tracks changes to our GitHub using a trigger and if accepted an image will be built. When training or sweeping hyperparameters on our local computer we write the result to weights and biases. The locally trained model is saved as a docker image but we also enable a setup for training the model using the trigger that tracks the GitHub repository and Vertex AI. The deployment of the fast-API is also saved as a docker image which is then uploaded to the GCP container registry. Furthermore, we have enabled two ways of deploying the model, both from the container registry from GCP and locally from the computer that an excited user would be able to use. ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- We started off with a model and project that was a little bit too complicated and we where struggling in the first few days to just get it running and especially to understand the structure of the image - caption dataset.
We originally wanted to fine tune a CLIP model, which as a multi-modal has more complexity than a standard text-to-text model.
So after spending too much time on the model, we reset our priorities on the MLOps part and took the simpler model that we ended up with, which still presented some challenges mainly in using all of these tools that were new to us. Going through the documentation, stackoverflow and chatGPT was however the winning strategy to overcome these problems. ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

---s220647: Set up git repo, set up gcp bucket, set up dvc, set up deployment in fast api and docker and cloud run, set up github workflows, set up hydra configuation set up cloudbuild, contributed in code in src.

---
