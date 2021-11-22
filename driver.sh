python3 ner/main.py \
    -f "baseline" \
    --batchsize 1024 \
    --nepochs 30 \
    --learningrate 0.1 \
    --embeddings "glove-wiki-gigaword-100" \
    --hiddenlayer 30 \
    --patience 10 \
    --bidirectional False

python3 ner/main.py \
    -f "large_dim_embedding" \
    --batchsize 1024 \
    --nepochs 30 \
    --learningrate 0.1 \
    --embeddings "glove-wiki-gigaword-300" \
    --hiddenlayer 30 \
    --patience 10 \
    --bidirectional False

python3 ner/main.py \
    -f "large_hidden_layer" \
    --batchsize 1024 \
    --nepochs 30 \
    --learningrate 0.1 \
    --embeddings "glove-wiki-gigaword-100" \
    --hiddenlayer 100 \
    --patience 10 \
    --bidirectional False

python3 ner/main.py \
    -f "bidirectional" \
    --batchsize 1024 \
    --nepochs 30 \
    --learningrate 0.1 \
    --embeddings "glove-wiki-gigaword-100" \
    --hiddenlayer 30 \
    --patience 10 \
    --bidirectional True