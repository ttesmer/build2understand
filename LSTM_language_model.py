#!/usr/bin/env python
# coding: utf-8
# %load_ext autoreload
# %autoreload 2
# %%:
import torch
import requests
import math
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
from collections import Counter

from rnns import LSTM

plt.rcParams["figure.figsize"] = [10, 20]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%:
try:
    open("datasets/nietzsche.txt", 'r')
except FileNotFoundError:
    with open("datasets/nietzsche.txt", "wb+") as f:
        response = requests.get('https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        f.write(response.content)
        f.close()

# %%:
with open("datasets/nietzsche.txt", 'r') as f:
    data = f.read().lower()
    data = data.replace("\n", " ")
    data = data[:10000]
    chars = sorted(list(set(data)))
    print(f"No. of chars: {len(chars)}\nLength of corpus: {len(data)}")
    f.close()

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

maxlen = 40
step = 3
# given input (X)
sentences = []
# next char to predict, target output (Y)
next_chars = []
for i in range(0, len(data) - maxlen, step):
    sentences.append(data[i : i + maxlen])
    next_chars.append(data[i + maxlen])

print(f"No. of sentences: {len(sentences)}")

# %%:
sample_sentence = "how could anything originate out of its opposite?"
#sample_sentence = "hello world hello world hello world hello world"
#sample_sentence = "abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz"
alphabet = sorted(list(set(sample_sentence)))

# %%:

print("Truth:")
print(sample_sentence)

seq_len = 40
NO_OF_EPOCHS = 4000

print(f"Epochs: {NO_OF_EPOCHS}\nSequence length: {seq_len}")

# %%:
if __name__ == "__main__":
    full_corpus = True

    if full_corpus:
        print("Runnin on full corpus..")
        print("Using random seed 25")
        torch.manual_seed(25)
        model = LSTM(
            device=DEVICE,
            epochs=NO_OF_EPOCHS,
            sequence_len=seq_len,
            alphabet_len=len(chars),
            hidden_size=128,
            lr=1e-2,
            beta=0.9,
            batch_size=128
        )
        model.fit(data, num_samples=len(sample_sentence)-1+200, print_loss=100)
    else:
        print("Only doing one sentence..")
        model = LSTM(
            device=DEVICE,
            epochs=NO_OF_EPOCHS,
            sequence_len=48,
            alphabet_len=len(alphabet),
            hidden_size=128,
            lr=1e-2,
            beta=0.90,
            batch_size=8
        )
        model.fit(sample_sentence, num_samples=len(sample_sentence)-1, print_loss=100)

# plt.plot(model.losses)
# plt.legend(["Training loss"])
# plt.show()
