from typing import List

from wasabi import msg
from tqdm import tqdm
import torch

from data import data_to_tensor


def train_model(
    model: torch.nn.Module,
    val_X: torch.Tensor,
    val_y: torch.Tensor, 
    n_epochs: int, 
    batches_tokens: List[List[str]],
    batches_tags: List[List[int]], 
    vocab: dict, 
    max_sentence_length: int, 
    optimizer: torch.optim.Optimizer, 
    patience: int, 
    mdl_fname: str
    ) -> torch.nn.Module:
    '''Trains a model with early stopping

    Parameters
    ----------
    model : torch.nn.Module
        initial model to train
    val_X : torch.Tensor
        validation data – input. Preprocessed with data.data_to_tensor()
    val_y : torch.Tensor
        validation data – labels. Preprocessed with data.data_to_tensor()
    n_epochs : int
        number of epochs to train for
    batches_tokens : List[List[str]]
        training data – input. Raw.
    batches_tags : List[List[int]]
        training data – labels. Raw.
    vocab : dict
        token to id mapping of words
    max_sentence_length : int
        n tokens in the longest sentence.
    optimizer : torch.optim.Optimizer
        optimizer to use for gradient descent
    patience : int
        number of times the model has to perform better than the last one in a row before stopping
    mdl_fname : str
        path to save the model

    Returns
    -------
    torch.nn.Module
        the best model trained
    '''

    # define empty list for validation losses and variable for best validation loss
    val_losses = []
    best_val_loss = None

    # for each epoch
    for epoch in tqdm(range(0, n_epochs)):
        msg.info(f'Epoch {epoch}')
        
        # convert training data and labels to tensors
        for token, label in zip(batches_tokens, batches_tags):
            X, y = data_to_tensor(
                tokens=token,
                labels=label,
                vocab=vocab,
                max_sentence_length=max_sentence_length,
            )

            # forward pass on training data
            y_hat = model.forward(X)

            # calculate loss
            loss = model.loss_fn(outputs=y_hat, labels=y)

            # backward propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # forward pass on validation data
        val_y_hat = model.forward(val_X)

        # compute loss on validation data
        val_loss = model.loss_fn(outputs=val_y_hat, labels=val_y)
        val_losses.append(val_loss)

        # early stoppping
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, mdl_fname)

        better = [vl for vl in val_losses[-patience:] if val_loss >= vl]
        if len(better) == patience:
            break

    # load best model
    model = torch.load(mdl_fname)

    return model
