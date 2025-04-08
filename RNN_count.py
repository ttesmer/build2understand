#!/usr/bin/env python
# coding: utf-8

# %%:
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from rnns import SimpleRNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%:
ss = "01234567890123456789012345678901234567890123456789"
print(f"Truth:\n{[*map(int,[*ss])]}")
print(len(set(ss)))

# %%:
seq_len=10
NO_OF_EPOCHS = 16

model = SimpleRNN(
    device=DEVICE,
    epochs=NO_OF_EPOCHS,
    sequence_len=seq_len,
    hidden_size=16,
    lr=1e-1,
    alphabet_len=len(set(ss))
)

model.fit(ss, num_samples=9)
count = '0'
count = model.predict(count, num_samples=9)
print('Count from zero:')
for c in count:
    print('-->', c)
#plt.plot(model.losses)
#plt.show()
