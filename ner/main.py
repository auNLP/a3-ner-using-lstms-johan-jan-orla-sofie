"""
This script uses a trainable LSTM and GloVe word embeddings to detect named entities in unstructured texts. 
The LSTM model is trained periodically using early stopping. Hence, if the model has not improves in N epochs, the training is stopped.
To evaluate the model, F1-score and accuracy score are computed.
"""

import argparse
import os
import random
import datetime

from wasabi import msg
import numpy as np
import gensim.downloader as api
from sklearn.metrics import classification_report
import torch
import torch.optim as optim

from data import batch, gensim_to_torch_embedding, load_data, data_to_tensor
from LSTM import TokenLSTM
from train import train_model


def main(
    mdl_fname: str,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    gensim_embedding: str,
    hidden_layer_dim: int,
    stopping_patience: int,
    bidirectional: bool
    ):
    '''
    Train & evaluate LSTM for NER labeling with early stopping.

    Parameters
    ----------
    mdl_fname : str
        path to save the model
    batch_size : int
        n datapoints in one batch
    n_epochs : int
        n epochs to train for
    learning_rate : float
        step size while descdending gradient
    gensim_embedding : str
        name of embedding model available in `gensim.api`
    hidden_layer_dim : int|list
        n nodes in hidden layer
    stopping_patience : int
        number of times the model has to perform better than the last one in a row before stopping
    bidirectional : bool
        run LSTM bidirectionally?
    '''

    # handle model filename for saving
    today = datetime.date.today()
    today_yymmdd = today.strftime("%y%m%d")
    mdl_fname = os.path.join('..', 'mdl_results', f'{today_yymmdd}_{mdl_fname}.ph')

    # set a seed to make the results reproducible
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # load data
    msg.info('Importing data')
    dataset = load_data()

    # shuffle training data
    train = dataset["train"].shuffle(seed=1)

    # lod validation data for early stopping 
    validation = dataset["validation"]

    # load test data for final validation
    test = dataset["test"]

    # load gensim word embeddings
    embeddings = api.load(gensim_embedding)

    # prepare data
    msg.info('Preparing data')
    
    # convert gensim word embeddings to torch word embeddings
    embedding_layer, vocab = gensim_to_torch_embedding(embeddings)

    # prepare training data
    # to get size of input layer, get length of the longest sentence
    max_train_len = max([len(s) for s in train['tokens']])
    max_val_len = max([len(s) for s in validation['tokens']])
    max_test_len = max([len(s) for s in test['tokens']])
    max_sentence_length = max([max_train_len, max_val_len, max_test_len])

    # prepare validation data (convert data and labels to tensors)
    val_X, val_y = data_to_tensor(
        validation["tokens"],
        validation["ner_tags"],
        vocab=vocab,
        max_sentence_length=max_sentence_length
        )
    
    # prepare test data (convert to tensors)
    test_X, test_y = data_to_tensor(
        test["tokens"],
        test["ner_tags"],
        vocab=vocab,
        max_sentence_length=max_sentence_length
        )

    # batch training tokens and labels 
    batches_tokens = batch(train["tokens"], batch_size)
    batches_tags = batch(train["ner_tags"], batch_size)

    # initialize the LSTM model
    LSTM = TokenLSTM(
        output_dim=10,
        embedding_layer=embedding_layer,
        hidden_dim_size=hidden_layer_dim,
        bidirectional=bidirectional
    )

    # initialize Adam optimizer
    optimizer = optim.Adam(
        params=LSTM.parameters(),
        lr=learning_rate)

    # train model with early stopping
    msg.info('Training LSTM model')
    m = train_model(
        model=LSTM,
        val_X=val_X,
        val_y=val_y,
        n_epochs=n_epochs,
        batches_tokens=batches_tokens,
        batches_tags=batches_tags,
        vocab=vocab,
        max_sentence_length=max_sentence_length,
        optimizer=optimizer,
        patience=stopping_patience,
        mdl_fname=mdl_fname
        )

    # evaluate model on test data
    msg.info('Evaluating performance')

    # calculate predictions for test data
    test_y_hat = m.forward(test_X)

    # flatten data
    test_true = test_y.view(-1)

    # reshape y_hat to be len(docs) x len(real labels)
    mask = torch.arange(0,9)
    test_pred = torch.index_select(test_y_hat, 1, mask)
    # get top label from predicted confidences
    test_pred = torch.argmax(test_pred, dim=1)
    # remove padding
    test_pred = [y_pred for y_pred in test_pred if y_pred != 1]
    test_true = [y for y in test_true if y != -1]
    assert len(test_true) == len(test_pred)

    test_true, test_pred = torch.tensor(test_true), torch.tensor(test_pred)

    # create clf report
    clf_report = classification_report(test_true, test_pred)
    
    # save model performance scores
    report_fname = mdl_fname.replace('.ph', '_report.txt')
    with open(report_fname, 'w') as f:
        f.write(clf_report)


if __name__ == "__main__":

    torch.device('cpu')

    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--mdlfname', type=str)
    ap.add_argument('--batchsize', type=int, default=1024)
    ap.add_argument('--nepochs', type=int, default=30)
    ap.add_argument('--learningrate', type=float, default=0.1)
    ap.add_argument('--embeddings', type=str, default='glove-wiki-gigaword-100')
    ap.add_argument('--hiddenlayer', type=int, default=30)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--bidirectional', type=bool, default=False)
    args = vars(ap.parse_args())

    main(
        mdl_fname=args['mdlfname'],
        batch_size=args['batchsize'],
        n_epochs=args['nepochs'],
        learning_rate=args['learningrate'],
        gensim_embedding=args['embeddings'],
        hidden_layer_dim=args['hiddenlayer'],
        stopping_patience=args['patience'],
        bidirectional=args['bidirectional']
    )
