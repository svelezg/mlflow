# ML Starter Repo

## Authors
* **Solution:** Santiago Vélez G. [svelez.velezgarcia@gmail.com](svelez.velezgarcia@gmail.com) [@svelezg](https://github.com/svelezg)
* **Problem statement:** Christian García


## Requirements
* Install MLflow and scikit-learn. There are two options for installing these dependencies:

   ** Install MLflow with extra dependencies, including scikit-learn (via pip install mlflow[extras])

   ** Install MLflow (via pip install mlflow) and install scikit-learn separately (via pip install scikit-learn)

* Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

* Clone (download) the repository via 
    
    git clone https://github.com/svelezg/mlflow.git


## Usage
### Training the Model
You can run the project from the directory using default path and mode as follows:

    python titanic/train.py

Or use other values for path and mode by passing them as arguments to train.py:

    python titanic/train.py <path> <mode>


### Comparing the Models
On the terminal

    mlflow ui

and view it at [http://localhost:5000]([http://localhost:5000])

### Run using conda environment
To run this project, invoke 

    mlflow run . -P path='./' mode='standard' 

After running this command, MLflow runs the training code in a new Conda environment with the 
dependencies specified in conda.yaml.


### Serving the Model

To deploy the server, run (replace the path with your model’s actual path):
    
    mlflow models serve -m ./mlruns/0/62e739c7071b48e2a3c675e20a1b372c/artifacts/model


Once you have deployed the server, you can pass it some sample data and see the predictions. 
The following example uses curl to send a JSON-serialized pandas DataFrame with the split orientation to the model server.

    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],"data":[[3.0, 1.0, 24.0, 0.0, 0.0, 8.0]]}' http://127.0.0.1:5000/invocations

the server should respond with output similar to:
```
    [{"0": 0.07294516265392303}]
```
# Motivation

Machine Learning has matured in the industry and like other areas it greatly benefits from having standardized procedures and best practices whenever possible. On the other hand, most of the available educational material on ML is about the theory but rarely are these best practices taught.


# Goal

In this project we will focus on one such best practice: building a suitable production-ready project structure for Machine Learning applications. 


## Values

*   **Reproducibility**: you should be able to reproduce any past experiment.
*   **Production Ready**: the models trained by your code should be able to easily be put into a production environment.
*   **Visibility**: you should be able to easily inspect the results, metrics, and parameters used for each experiment.
*   **Generality**: the code base should serve as a template for future machine learning projects
*   **Speed**: on your next project the template should drastically cut your time to production.


## Objectives

1. Train a model for the [Titanic Dataset](https://www.kaggle.com/c/titanic/data). The model will not be your focus but rather an excuse to create the project structure.
2. Your training code should be able to take command line arguments so it's easily usable from bash. Important parameters are:
    1. data_path: Input data should not be a constant since the repo should be general.
    2. debug: (optional) whether you are in debug mode.
    3. model_type: (optional) you can support changing the model used for training.
3. On each run / experiment your code should do the following tasks:
    4. Serialize/store the exact input parameters used for the experiment
    5. Serialize/store the resulting _metrics_ from experiment.
    6. Serialize/store the trained model plus the exact preprocessing procedure such that inference can be made **without** the original codebase. [pickle, model.save]
        1. Train -> save model to storage -> load model in sever
    7. Your code should serialize/store the exact code used in the experiment.
4. At the end of the project create a separate repo with the same code and remove any project specific parts, add comments of where the next user should probably insert important code. Create a README telling users how to easily use the template.


### Bonus
*   Try to have a way to visualize the parameters and metrics given by various experiments so you can compare them.
*   Try to separate code that sets up the experiment (which should be generic) from the code that does the preprocessing procedure and model definition (which is project specific).
*   Add nice features to the template such as data splitting, automatic exploration of the data, hyper parameter tuning, etc. 


### Tips

*   Check out tools / services like [ML Flow](https://mlflow.org/) and [Weights and Biases](https://www.wandb.com/).
*   [Scikit Learn’s custom transformers](https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156) are a great way to perform complex preprocessing.
*   Structure idea

    /src # your actual code

       ….
    /results  #&lt;- gitignore
       experiment1/ # serialized stuff goes here
           …
       experiment2/
