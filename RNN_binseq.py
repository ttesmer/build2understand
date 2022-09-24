#!/usr/bin/env python
# coding: utf-8

# %%:
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from rnns import SimpleRNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%:
ss = "1010101010101010101010101010101010101010"
print(f"Truth:\n{[*map(int,[*ss])]}")

# %%:
seq_len=10
NO_OF_EPOCHS = 5

model = SimpleRNN(
    device=DEVICE,
    epochs=NO_OF_EPOCHS,
    sequence_len=seq_len,
    hidden_size=16,
    lr=1e-1,
    alphabet_len=2
)

model.fit(ss, num_samples=40)
plt.plot(model.losses)
plt.show()
