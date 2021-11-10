import numpy as np
import torch
import random

import gensim.downloader as api

from ner.data import batch, gensim_to_torch_embedding, load_data
from ner.LSTM import TokenLSTM


def main(gensim_embedding: str, batch_size: int, epochs: int, learning_rate: float, patience: int=10):
    # set a seed to make the results reproducible
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset = load_data()
    train = dataset["train"]

    embeddings = api.load(gensim_embedding)

    # convert gensim word embedding to torch word embedding
    embedding_layer, vocab = gensim_to_torch_embedding(embeddings)


    # Preparing data
        # shuffle dataset
    train = dataset["train"].shuffle(seed=1)

    # batch it using a utility function (don't spend time on the function, but make sure you understand the output)
    batches_tokens = batch(train["tokens"], batch_size)
    batches_tags = batch(train["ner_tags"], batch_size)

    # Initialize the model
    # Initialize optimizer

    # Train model (I suggest writing a function for this)
        ## for each epoch
        ## for each batch
        
        ## prepare data (see code from the class on RNNs)
        
        ## train on one batch
        # run forward pass
        # run backward pass
        # update parameters
        # calculate loss

        ##  periodically calculate loss on validation set
        # if epoch % 10 == 0: # e.g. every 10 epochs
            # save the model if it is the best so far
            # stop the training if you haven't saved a better model in the last N epochs (this N here is typically referred to as patience)

    # Load the best model
    # Calculate the Accuracy and F1 on the test set (It might be easier to write a function for this)
    