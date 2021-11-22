from collections import Counter
from ner.data import data_to_tensor, load_data

def create_tags_to_ids():
    dataset = load_data()
    tags = dataset["train"].features["ner_tags"].feature.names
    tags_to_ids = {tag: i for i, tag in enumerate(tags)}
    return tags_to_ids


def test_data_to_tensor():
    """test that the prepare batch function outputs the correct shape"""
    sample_texts = [
        ["I", "am", "happy"],
        ["I", "like", "to", "eat", "pizza"],
        ["Anders", "like", "to", "eat", "pizza"],
    ]
    sample_labels = [
        ["O", "O", "O"],
        ["O", "O", "O", "O", "O"],
        ["I-PER", "O", "O", "O", "O"],
    ]

    max_sentence_length = max([len(doc) for doc in sample_texts])
    vocab = dict()
    for doc in sample_texts:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    vocab.update({'UNK': len(vocab)})
    vocab.update({'PAD': len(vocab)})

    tags_to_ids = create_tags_to_ids()
    sample_labels = [[tags_to_ids[tag] for tag in tags] for tags in sample_labels]

    X, y = data_to_tensor(sample_texts, sample_labels, vocab, max_sentence_length)
    
    assert X.shape == (3, 5), "Your prepared batch does not have the correct size"
    assert all(i in y.unique() for i in [-1, 0, 2]), "Your prepared batch does not contain the correct labels"
