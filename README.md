# Word-Embedding-PyTorch

**`Word Embeddings` are a representation of the *`semantics of a word`*, efficiently encoding semantic information that might be relevant to the task at hand.**

- Embeddings in pytorch are like a lookup table that stores the word vectors. It simply returns the word vector for the given word (index).
- In simplest words, it is a 2D tensor of shape (vocab_size, embedding_dim).
- For the word at index `i`, the embedding is the `i-th` row of the matrix.
- It is a trainable layer in the neural network. (`a sparse one-hot vector is multiplied with the embedding matrix to get the word vector`)

![Word Embedding](./assets/Visualization-of-the-word-embedding-space.png)

- Converting Words to **vector** form, that contains the semantic information of the word.
- This Embedding can be used as the input to the model, to perform the task like `Sentiment Analysis`, `Named Entity Recognition`, `Part of Speech Tagging`, etc.
- PyTorch provides a module `nn.Embedding` to create and use Word Embeddings.
- It can be trained on the large corpus of text, or can be used as the pre-trained model.

---

## Training Word Embeddings

1. N-Gram
2. Skip-Gram
3. Continuous Bag of Words (CBOW)
4. Using Pre-trained Word Embeddings (GloVe, Word2Vec, FastText)

---

## N-Gram

- N-Gram is a contiguous sequence of `n` items from a given sample of text or speech.

![N-Gram](./assets/n-gram.jpg)
```markdown
sentence = "The quick brown fox jumps over the lazy dog"

# Unigram
[The, quick, brown, fox, jumps, over, the, lazy, dog]

# Bigram
[The quick, quick brown, brown fox, fox jumps, jumps over, over the, the lazy, lazy dog]

# Trigram
[The quick brown, quick brown fox, brown fox jumps, fox jumps over, jumps over the, over the lazy, the lazy dog]

# --------------------

## Input & Output

input | output
-------|-------
The quick | brown
quick brown | fox
brown fox | jumps
fox jumps | over
jumps over | the
over the | lazy
the lazy | dog
```

---

## Skip-Gram

- Skip-gram is a method to **predict the context words from the target words**.

- **one input word** is used to **predict the surrounding words**.

![Skip-Gram](./assets/skip-gram.png)

```markdown
# Example
sentence = "The quick brown fox jumps over the lazy dog"

# Input & Output
input   |    output
------- | ----------------------------
brown   | (quick The) (fox jumps)
fox     | (brown quick) (jumps over)
...
```

---

## Continuous Bag of Words (CBOW)

- CBOW is a method to **predict the target words from the surrounding context words (words before and after)**.

![CBOW](./assets/continuous-bag-of-words.png)

```markdown
# Example
sentence = "The quick brown fox jumps over the lazy dog"

# Input & Output
input | output
-------|-------
(quick The) (fox jumps) | brown
(brown quick) (jumps over) | fox
...
```

---

## Using Pre-trained Word Embeddings (GloVe, Word2Vec, FastText) 😎

- We can use the pre-trained word embeddings like `GloVe`, `Word2Vec`, `FastText` to get the word vectors. And then use them as the initial weights of the embedding layer. (Transfer Learning)
- **We can choose to keep these weights fixed or train them further on our dataset. (Fine-tuning)**

```python
# FloatTensor containing pretrained weights
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])

embedding = nn.Embedding.from_pretrained(weight)
# Get embeddings for index 1
input = torch.LongTensor([1])
embedding(input)
```

- To use the pre-trained word embeddings, we can use the `torchtext` library, which provides the `pre-trained word vectors` and `tokenization` methods.
- We can also use the `gensim` library to load the pre-trained word vectors.

---

## **Using Gensim to clean, tokenize, and create the word vectors using FastText model**

- Gensim & FastText:
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html#sphx-glr-auto-examples-tutorials-run-fasttext-py

```markdown
Here's the workflow using Gensim:

**1. Preprocessing and Text Cleaning (Optional):**

* Use Gensim's functionalities like `gensim.parsing.preprocessing` to clean and pre-process your text data. This can involve steps like removing stop words, stemming/lemmatization, and handling special characters.

**2. Training Word Embeddings with Gensim (Optional):**

* Instead of using the standalone FastText tool, utilize Gensim's `gensim.models.FastText` class to train the word embeddings directly within your Python environment. This eliminates the need for separate tools and enables easier integration with your PyTorch model.
* Define the parameters of the FastText model like vector size, window size, etc., and train the model on your pre-processed text corpus.

**3. Saving the Trained Model:**

* After training, save the model using methods like `model.save()`. This saves the learned word vectors and related information.

**4. Loading Embeddings in PyTorch:**

* Use Gensim's functionalities or PyTorch libraries like `torchtext` to load the saved model and access the word vectors.

**5. Create and Initialize Embedding Layer:**

* Create an embedding layer in PyTorch with the same dimension as the loaded vectors.
* **Here's the key difference:** Instead of directly copying the pre-trained vectors, use them as the **initial weights** for the embedding layer. PyTorch allows you to define the initial weights of layers.
* This leverages the pre-trained information while allowing the model to fine-tune the weights during training, adapting to your specific data.

**6. Train your PyTorch model:**

* Train your PyTorch model as usual, allowing the embedding layer weights to be updated based on your training data.

This workflow utilizes Gensim for both pre-processing and training (optional) while leveraging PyTorch for model building and fine-tuning.
```

---

## **Using TorchText to load pre-trained word vectors and tokenize the text data**

https://pytorch.org/text/stable/vocab.html

```markdown
- Check for `GloVe`, `FastText`, etc, in the `torchtext.vocab` module.
```

- Read here for the code: https://github.com/pytorch/text/issues/1350

```python
from torchtext.vocab import GloVe, vocab
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
import torch
import torch.nn as nn

#define your model that accepts pretrained embeddings 
class TextClassificationModel(nn.Module):

    def __init__(self, pretrained_embeddings, num_class, freeze_embeddings = False):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings, freeze = freeze_embeddings, sparse=True)
        self.fc = nn.Linear(pretrained_embeddings.shape[1], num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

train_iter = AG_NEWS(split = 'train')
num_class = len(set([label for (label, _) in train_iter]))
unk_token = "<unk>"
unk_index = 0
glove_vectors = GloVe()
glove_vocab = vocab(glove_vectors.stoi)
glove_vocab.insert_token("<unk>",unk_index)
#this is necessary otherwise it will throw runtime error if OOV token is queried 
glove_vocab.set_default_index(unk_index)
pretrained_embeddings = glove_vectors.vectors
pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))

#instantiate model with pre-trained glove vectors
glove_model = TextClassificationModel(pretrained_embeddings, num_class)


tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split = 'train')
example_text = next(train_iter)[1]
tokens = tokenizer(example_text)
indices = glove_vocab(tokens)
text_input = torch.tensor(indices)
offset_input = torch.tensor([0])

model_output = glove_model(text_input, offset_input)
```