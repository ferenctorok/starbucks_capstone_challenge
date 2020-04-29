### Udacity Machine Learning Engineer Nanodegree Program ###

# Starbuck's Capstone Project #

This is the final project of the [Udacity Machine Learning Engineer Nanodegree Program](https://www.udacity.com/coursemachine-learning-engineer-nanodegree--nd009t).

The dataset contains simulated data that mimics costumer behavior. It consists of personal data, offers, transactions with timestamp and activities with offers with timestamp. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free).

Not all users receive the same offer, and that is the challenge to solve with this data set.

The task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks in reality has a wild variety of products. 

The aim of our solution was to categorize received offers into the following outcome categories:
* Will not be viewed nor completed accidentally
* Will not be viewed but will be completed
* Will be viewed but will not be completed
* Will be completed and viewed

For this we combine data from the whole dataset to create features which sufficiently represent the person's costumer history until the offer was received. For this purpose a Feed Forward Neural Network was trained.

### included in the repository
* `proposal.pdf` : Capstone Proposal
* `report.pdf` : report of the Capstone Project
* `data_exploration.ipynb` : notebook for data exploration and visualization
* `feature_engineering.ipynb` : notebook for feature engineering
* `training.ipynb` : notebook containing the training code.
* `source` : directory containing .py script files which where imported in the notebooks.
* `data` : directory containing the original and the produced training data.
* `cache` : directory containing some cache files.
* `models` : directory containing the state dictionary of the trained models.
* `latex_source` : directory containing the latex source files.
* `requirements.txt` : for installation with conda.

### required packages:
All the required packages can be installed using conda from `requirements.txt` by running this command on linux:

`conda create --name <env> --file requirements.txt`

packages:
* jupyter 1.0.0
* numpy 1.18.1
* pandas 1.0.3
* matplotlib 3.1.3
* os 0.1.4
* scikit-learn 0.22.1
* importlib 1.6.0
* pytorch 1.5.0