from ner.main import main

def test_main():
    """test that main run using a single epoch"""

    main(
        mdl_fname='test',
        gensim_embedding="glove-wiki-gigaword-50",
        n_epochs=1,
        batch_size=5,
        learning_rate=0.1,
        hidden_layer_dim=6,
        stopping_patience=1,
        bidirectional=False 
        )

