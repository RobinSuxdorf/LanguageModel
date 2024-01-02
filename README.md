# Language Model Project

Welcome to my Language Model project! This repository contains my own custom language model, built with a focus on an encoder-only architecture that utilizes the attention mechanism. I've learned a lot from the tutorials by Andrej Karpathy and have been inspired by the Llama source code. Unfortunately, more data and training is necessary for good results.

## Installation

For installing the needed packages use the following command:

```
pip install -r requirements.txt
```

The model can be downloaded [here](https://drive.google.com/file/d/1xW2tuO7pIwbCJcfgBGHX6zyu0RXAC-nD/view?usp=share_link).
## Usage

First, we need to define the device on which we want to run the model:

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Next, we can load the model as following:

```python
from language_model import generation

model = generation.LanguageModel.load('model.pt', device=device)
```

Finally, we can use the language model:

```python
model.predict('What is Deep Learning?')
```

## Train

The Language model is trained on one GPU with 10GB memory. To prevent OutOfMemoryError gradient accumulation was used with 4 accumulation steps before updating the weights.

## Hyperparameters

| Hyperparameter | Batch Size | Learning Rate | Epochs | Context Length |
| -------------- | ---------- | ------------- | ------ | -------------- |
| Model-30M      | 16         | 3e-4          | 20     | 1024           |
## Next Ideas

The primary goals for this project include:

1. **Word Embeddings**: Extract word embedding from the language model.

2. **Tokenizer in C++**: Developing a fast and efficient text tokenizer in C++ to accelerate text processing, which is particularly important for large-scale text generation tasks.

3. **Storing the hyperparameters in config file**: Currently, the hyperparameters are defined in two differnet places: TrainArgs and ModelArgs. It might be more convenient to define them inside the same file.
