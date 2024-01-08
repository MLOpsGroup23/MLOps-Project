# MLOps_Project

Exam project for the MLOps course at DTU

## Project description
### Goal
We will use various machine learning DevOps tools to create and test models for classification of the Fashion MNIST dataset. We have chosen a well known dataset, which should be easily solveable, as we want to spent the majority of the time on setting up proper organisation, reproduceability, profiling, logging, continous integration and similar Dev-Ops principles.
### Framework 
We plan to use [Pytorch Lightning](https://lightning.ai/). Furthermore, we want to use Hydra in order to store and use several configurations easily, which makes managing experiments easier. We want to integrate Weights and Biases to log performance of different models, such that comparisons can easily be made in the future. 
### Data
The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is a dataset consisting of 28x28 greyscale images belonging to one of 10 classes. The dataset is already split into a training set of 60,000 images and a test set of 10,000 images. The Fashion MNIST is part of [Pytorch datasets](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html), which makes it easy to load and use.
### Model 
Initially, we will use [ResNet](https://pytorch.org/vision/main/models/resnet.html) as a baseline model and then we will experiment with other models. For example using VAE encoders with a linear classification head.  


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── MLOps_Project  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
