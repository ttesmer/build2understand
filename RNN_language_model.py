#!/usr/bin/env python
# coding: utf-8

# %%:
import torch
import requests
import math
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
from collections import Counter

from rnns import rnn

plt.rcParams["figure.figsize"] = [10, 5]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%:
try:
    open("datasets/nietzsche.txt", 'r')
except FileNotFoundError:
    with open("datasets/nietzsche.txt", "wb+") as f:
        response = requests.get('https://s3.amazonaws.com/text-datasets/datasets/nietzsche.txt')
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

seq_len = 48
NO_OF_EPOCHS = 4000

print(f"Epochs: {NO_OF_EPOCHS}\nSequence length: {seq_len}")

# %%:
if __name__ == "__main__":
    model = rnn.SimpleRNN(
        device=DEVICE,
        epochs=NO_OF_EPOCHS,
        sequence_len=seq_len,
        alphabet_len=len(alphabet),
        hidden_size=100,
        lr=1e-1,
        batch_size=1#math.ceil(len(data)/seq_len)
    )
    try:
        model.fit(sample_sentence, num_samples=200, print_loss=100)
    except KeyboardInterrupt:
        print("Interrupted, saving parameters..")
        model.save("model_params")

plt.plot(model.losses)
plt.legend(["Training loss"])
plt.show()
