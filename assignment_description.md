

# Assignment 3: Named entity recognition using LSTMs
This is the third assignment, where the goal is to use LSTMs and word embeddings for detecting entities in unstructured texts.


You are given:
- A trainable LSTM model using word embeddings. For classifying the entities of an input text.
- 

You will need to:

- [ ] A `prepare_batch` function, which prepares a batch of inputs for the LSTM. *Hint* examine the starter code from class 8. An outline for this function is present in `data.py`. If this takes a long time you can always save the processed dataset and read it in.
- [ ] Train an LSTM model trained for English NER using the conllpp dataset. This should include three experiments
  - [ ] One comparing the effect of the word embedding size, you can see available word embedding on gensim [here](https://github.com/RaRe-Technologies/gensim-data).
  - [ ] And two others which you select yourself, some ideas could be:
    - Compare the effect of the hidden layers size of the LSTM
    - Compare a bidirectional LSTM with your unidirectional LSTM (You can do this by setting the `bidirectional=True`)
    - Compare the effect of different word embeddings of similar size (e.g. trained on different domains)
    - Compare the `nn.RNN` as opposed to the `nn.LSTM` block
    - Compare the effect of different optimizers
    - ...
  - [ ] Your training loop should periodically be applied to the validation set. If the model performs better save it. You can save the model using `torch.save` and load it using `torch.load`.
    - [ ] Early stopping: If the model haven't improved in over N epochs (e.g. 10 epochs), stop the training. 
  - [ ] Using the best performing model calculate the (micro) F1 score and accuracy of the model on the test set. Feel free to use the sklearn implementation of the [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) score and accuracy.
  - [ ] Fill out the readme
- OPTIONAL:
  - [ ] It is quite normal to use more than one embedding at the same time (by concatenating them). Does this increase your performance?
  - [ ] Train an LSTM model trained for sentiment classification. Here you will need to modify the existing LSTM a good idea is to start of by examining the documentation for the LSTM module. Here you can use the `load_sst2` function supplied. I have created an outline for how this would work in the `SentenceLSTM` in `LSTM.py`.


*Note*: Naturally, the pre-implemented tests should pass and that you are welcome to add more tests. Please also tick off the boxes if you have completed the task.


## Project Organization
The organization of the project is as follows

```
├── LICENSE                <- the license of this code
├── README.md              <- The top-level README for this project.
├── .github            
│   └── workflows          <- workflows to automatically run when code is pushed
│   │    └── pytest.yml    <- A workflow which runs pytests upon push
├── ner                    <- The main folder for scripts
│   ├── tests              <- The pytest test suite
│   │   └── ...
|   └── ...
├── .gitignore             <- A list of files not uploaded to git
└── requirements.txt       <- A requirements file of the required packages.
```


## Intended learning goals
- Being able to work with recurrent layers using PyTorch.
- Understanding of named entity recognition including the structure of its labels and how a model could be trained.
- Being able to conduct meaningful experiments that influence the performance of the model.
- Being able to implement a simple version of early stopping.


## FAQ

<br /> 

<details>
  <summary> Pytest: How do I test the code and run the test suite?</summary>

To run the test suite (pytests) you will need to install the required dependencies. This can be done using 


```
pip install -r requirements.txt
pip install pytest

python -m pytest
```

which will run all the test in the `tests` folder.

Specific tests can be run using:

```
python -m pytest path/to/test_script.py
```

**VS Code**
You can also run your test directly in VS Code. See the guide on the [pytest integration](https://code.visualstudio.com/docs/python/testing) here.

**Code Coverage**
If you want to check code coverage you can run the following:
```
pip install pytest-cov

python -m pytest --cov=.
```



</details>


<br /> 
