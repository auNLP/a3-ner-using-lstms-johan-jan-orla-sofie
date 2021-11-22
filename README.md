<h1 align="center"> Assignment 3: Named Entity Recognition (NER) <br> Using LSTMs and Word Embeddings to detect entities in unstructured texts</h1>

<p align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/company.png" width="75" height="75"/>
</p>

## Table of Contents
- Description
- Learning goals
- Results
- Reproducing the Scripts
- Project Organization

## Assignment Description <br> 
In this assignment, a recurrent neural network in the form of a Long Short Term Memory (LSTM) model was trained to identify named entities in the 'CoNLLPP' dataset using gensim's GloVe word embeddings. The LSTM model was trained periodically using early stopping. Hence, if the model has not improves in a specified number of epochs, the training is stopped and the model is saved.The CoNLLPP dataset is an improvement of the popular CoNLL-2003 dataset, which contains sentences collected from the Reuters Corpus. This corpus consists of Reuters news stories between August 1996 and August 1997. For the training and development set, ten days worth of data were taken from the files representing the end of August 1996. For the test set, the texts were from December 1996. The preprocessed raw data covers the month of September 1996.
To evaluate the model, F1-score and accuracy score were computed. Moreover, three experiments were conducted:
1. Comparing the effect of the word embedding size
2. Comparing the effect of the size of the hidden layer in the LSTM model
3. Comparing the use of a bi-directional LSTM with a unidirectional LSTM

**What are LSTM Models?** <br>
LSTM models are a special type of recurrent neural network capable of learning order dependence in sequence prediction problems. This gives them the ability to learn long term dependencies, where both information before and later in the sentence can be used to inform of a word's nature or importance. They are especially good for named entity tasks where a deeper unstanding of the contextual information is needed. 

## Learning goals of the assignment
1. To work with recurrent layers using PyTorch 
2. To understand the nature of named entity recognition tasks  
3. To be able to implement early stopping and meaninful experiments which influence the performance of the model

## Results
The results of the LSTM model with default parameters (n_epochs = 30, lr = 0.01, hidden_dim = 30, patience = 10, optimizer = Adam, bidirectional = False, word_embedding_dim = 100-dim GloVe) and the experiments which include changing the dimensions of the word embeddings (exp. 1), changing the size of the hidden layer (exp. 2) and changing the LSTM to bidirectional (exp. 3) are reported below:

|               | Default model parameters | Exp. 1: word embeddings = 300 dim | Exp. 2: Hidden layer dim = 100 | Exp. 3: Bidirectional = True |
|---------------|---------|-----------------------------------|--------------------------------|------------------------------|
| Accuracy      | 0.82    | 0.82                              | 0.82                           | 0.87                         |
| Macro avg. F1 | 0.12    | 0.13                              | 0.13                           | 0.32                         |

The experiments in which we change the word embedding dimensions and the hidden layer size do not change the results. However, when we make the LSTM model bidirectional rather than unidirectional, we see notable improvements in both the accuracy and the F1-score. We still get very low F1-score across all labels, indicating that the model is overfitting to a single class (i.e. 0). 

## Reproducing the Scripts 
1. If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. First, the user will have to create their own version of the repository by cloning it from GitHub. This is done by executing the following from the command line: 

```
$ git clone https://github.com/auNLP/a3-ner-using-lstms-johan-jan-orla-sofie.git named-entity-recognition
```

2. Once the user has cloned the repository, a virtual environment must be set up in which the relevant dependencies can be installed. To set up the virtual environment and install the relevant dependencies, a bash-script is provided, which automatically creates and installs the dependencies listed in the ```requirements.txt``` file when executed. To run the bash-script that sets up the virtual environment and installs the relevant dependencies, the user must execute the following from the command line. 

```
$ cd named-entity-recognition
$ bash create_venv.sh 
```

3. Once the virtual environment has been set up and the relevant dependencies listed in the ```requirements.txt``` have been installed within it, the user is now able to run the ```main.py```script from the command line. In order to run the script, the user must first activate the virtual environment in which the script can be run. Activating the virtual environment is done as follows.

```
$ source named-entity-venv/bin/activate
```

4. Once the virtual environment has been activated, the user is now able to run the ```main.py```script.

```
(named-entity-venv) $ python main.py --epochs 10 --gensim_embedding glove-wiki-gigaword-100
```

## Project Organization
The organization of the project is as follows:

```
├── LICENSE                    <- the license of this code
├── README.md                  <- The top-level README for this project.
├── .github            
│   └── workflows              <- workflows to automatically run when code is pushed
│   │    └── pytest.yml        <- A workflow which runs pytests upon push
├── mdl_results                <- Model results 
├── ner                        <- The main folder for scripts
│   ├── tests                  <- The pytest test suite
│   │   └── ...
|   └── ...
├── .gitignore                 <- A list of files not uploaded to git
├── requirements.txt           <- A requirements file of the required packages.
└── assignment_description.md  <- the assignment description
```