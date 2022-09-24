#!/usr/bin/env python
# coding: utf-8

# %%:
import torch
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow

from rnns import LSTM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%:
sequence = "1010101010101010101010101010101010101010"
print(f"Truth:\n{sequence}")
alphabet = sorted(list(set(sequence)))

# %%:
NO_OF_EPOCHS=20
HIDDEN_SIZE=16

model = LSTM(
    device=DEVICE,
    # phase out putting epochs here.. should be in optimizer or model.fit()
    epochs=NO_OF_EPOCHS,
    hidden_size=HIDDEN_SIZE,
    sequence_len=len(sequence),
    alphabet_len=len(alphabet),
    lr=0.1,
    beta=0.9,    
)

X = torch.zeros((len(sequence)-1, len(alphabet), 1), device=DEVICE)
Y = torch.zeros_like(X)
char2idx = {c: i for i,c in enumerate(alphabet)}
idx2char = {i: c for i,c in enumerate(alphabet)}
model.idx2char = idx2char

for t in range(len(sequence)-1):
    char = sequence[t]
    next_char = sequence[t+1]
    X[t][char2idx[char]] = 1.
    Y[t][char2idx[next_char]] = 1.

print("Using random seed 25")
model.fit_no_SGD(X, Y, num_iters=NO_OF_EPOCHS, weight_decay=1e-3, print_loss=5)
