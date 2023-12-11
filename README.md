# Language Model Project

Welcome to my Language Model project! This repository contains my own custom language model, built with a focus on an encoder-only architecture that utilizes the attention mechanism. I've learned a lot from the tutorials by Andrej Karpathy and have been inspired by the Llama source code.

## Installation

For installing the needed packages use the following command:
```
pip install -r requirements.txt
```
## Next Ideas

The primary goals for this project include:

1. **End-of-Sentence (EOS) Tokens**: Implementing end-of-sentence tokens to improve the model's understanding of sentence boundaries and improve text generation.

2. **Word Embeddings**: Extract word embedding from the language model.

3. **Tokenizer in C++**: Developing a fast and efficient text tokenizer in C++ to accelerate text processing, which is particularly important for large-scale text generation tasks.

4. **Model takes care about input token length**: The model checks whether the number of tokens of the input text is smaller than the context length.
