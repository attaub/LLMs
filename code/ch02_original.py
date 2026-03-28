
from importlib.metadata import version

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))


# ## 2.1 Understanding word embeddings

# ## 2.2 Tokenizing text

# tokenize text 

import os
import requests

# if not os.path.exists("the-verdict.txt"):
#     url = (
#         "https://raw.githubusercontent.com/rasbt/"
#         "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#         "the-verdict.txt"
#     )
#     file_path = "the-verdict.txt"

#     response = requests.get(url, timeout=30)
#     response.raise_for_status()
#     with open(file_path, "wb") as f:
#         f.write(response.content)


with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

import re

def print_text(text_str):
    for ix, word in enumerate(text_str):
        print(f"{ix + 1}-->{word}")

text = "Hello, world. This, is a test."

result = re.split(r'(\s)', text)
print(result)
print_text(result)

result = re.split(r'([,.]|\s)', text)
print(result)
print_text(result)


result = [item for item in result if item.strip()]
print(result)
print_text(result)


text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print_text(preprocessed[:30])


print(len(preprocessed))


# ## 2.3 Converting tokens into token IDs

# Convert the text tokens into token IDs that we can process via embedding layers later

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

# The first 50 entries in this vocabulary:

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 40:
        break

# SimpleTokenizerV1


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)


tokenizer.decode(ids)
tokenizer.decode(tokenizer.encode(text))


# ## 2.4 Adding special context tokens

# special tokens for unknown words and to denote the end of a text

# [BOS] beginning of sequence
# [EOS] end of sequence
# [PAD] Padding
# [UNK] 
# <|endoftext|> is analogous to the `[EOS]` token mentioned above
# GPT also uses the <|endoftext|> for padding 
# GPT-2 does not use an <UNK> token, it uses a byte-pair encoding (BPE) tokenizer
# Use the <|endoftext|> tokens between two independent sources of text


tokenizer = SimpleTokenizerV1(vocab)

text = "Hello, do you like tea. Is this-- a test?"


try:
    tokenizer.encode(text)
except ValueError:
    print("Unknown token encounterd")


# The above produces an error because the word "Hello" is not contained in the vocabulary


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
len(vocab.items())

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# adjust the tokenizer accordingly so that it knows when and how to use the new `<unk>` token

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# Lets try to tokenize text with the modified tokenizer:

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer.encode(text)
tokenizer.decode(tokenizer.encode(text))

# ## 2.5 BytePair encoding
# allows the model to break down words that aren't in its predefined vocabulary 

import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)


# ## 2.6 Data sampling with a sliding window

with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")


# One by one, the prediction would look like as follows:

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# Install and import PyTorch 

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# ## 2.7 Creating token embeddings

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)


# The embedding layer approach above is essentially just a more efficient way of implementing one-hot encoding followed by matrix multiplication in a fully-connected layer, which is described in the supplementary code in [./embedding_vs_matmul](../03_bonus_embedding-vs-matmul)

print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))

# An embedding layer is essentially a look-up operation:

# ## 2.8 Encoding word positions

# Embedding layer convert IDs into identical vector representations regardless of where they are located in the input sequence:


# Positional embeddings are combined with the token embedding vector to form the input embeddings for a large language model:

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)


print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)


token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# print(token_embeddings)


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
print(pos_embedding_layer.weight)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)
print(pos_embeddings)

# Create the input embeddings used in an LLM, we simply add the token and the positional embeddings

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

# print(input_embeddings)

