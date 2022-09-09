import torch
from torch import matmul
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

class SimpleRNN(object):
    def __init__(self,
                 device,
                 epochs: int,
                 sequence_len: int,
                 hidden_size: int,
                 alphabet_len: int,
                 lr: float,
                 batch_size=1):
        super().__init__()
        """
        Basic Recurrent Neural Network.
        """
        self.device = device
        self.epochs = epochs
        self.seq_len = sequence_len
        self.hidden_size = hidden_size
        self.alphabet_len = alphabet_len
        self.batch_size = batch_size
        self.eta = lr

        # input-hidden
        self.U = torch.randn(hidden_size, alphabet_len, device=device) * 1e-2
        self.U.grad = torch.zeros_like(self.U)
        self.memU = torch.zeros_like(self.U)
        # hidden-hidden
        self.W = torch.randn(hidden_size, hidden_size, device=device) * 1e-2
        self.W.grad = torch.zeros_like(self.W)
        self.memW = torch.zeros_like(self.W)
        # hidden-output
        self.V = torch.randn(alphabet_len, hidden_size, device=device) * 1e-2
        self.V.grad = torch.zeros_like(self.V)
        self.memV = torch.zeros_like(self.V)

        # activation bias
        self.b = torch.zeros(hidden_size, 1, device=device)
        self.b.grad = torch.zeros_like(self.b)
        self.memb = torch.zeros_like(self.b)
        # output bias
        self.c = torch.zeros(alphabet_len, 1, device=device)
        self.c.grad = torch.zeros_like(self.c)
        self.memc = torch.zeros_like(self.c)

        # hidden states
        self.hs = {}
        # losses
        self.losses = []

    def fit(self, data, num_samples, print_loss=1):
        h_last = torch.zeros(self.hidden_size, 1, device=self.device)
        pointer = 0
        alphabet = sorted(list(set(data)))
        char2idx = {c: i for i,c in enumerate(alphabet)}
        idx2char = {i: c for i,c in enumerate(alphabet)}
        self.char2idx = char2idx
        self.idx2char = idx2char
        for epoch in range(self.epochs):
            # Reset gradient & hidden state
            self.zero_gradient()
            self.clear_state()
            # Reset pointer to not go out of index bounds
            if pointer+self.seq_len+1 >= len(data):
                pointer = 0
                h_last = torch.zeros(self.hidden_size, 1, device=self.device)
                self.hs[-1] = h_last

            # Get next part of sequence & its labels
            inputs = [char2idx[char] for char in data[pointer:pointer+self.seq_len]]
            targets = [char2idx[char] for char in data[pointer+1:pointer+self.seq_len+1]]
            print(f"inputs: {data[pointer:pointer+self.seq_len]}")
            assert len(inputs) == len(targets), f"len inputs: {len(inputs)} doesn't match len of targets: {len(targets)}"
            assert len(inputs) == self.seq_len, f"len inputs: {len(inputs)} doesn't match seq_len: {self.seq_len}"


            X = torch.zeros((len(inputs), self.alphabet_len, 1), device=self.device)
            Y = torch.zeros_like(X)
            preds = torch.zeros_like(X)
            self.hs[-1] = h_last
            for t in range(len(inputs)):
                # Change index of input and target characters
                X[t][inputs[t]] = 1.0
                Y[t][targets[t]] = 1.0

                # Make sure the shape is a matrix
                assert X[t].shape == torch.Size([self.alphabet_len, 1]), f"X[t] wrong shape: {X[t].shape}"

                # Get softmax probabilities and last state
                y_hat, h_last = self.forward(X[t], h_last)
                # Add hidden state to correct time index
                self.hs[t] = h_last
                # Make sure the shape is a matrix
                assert y_hat.shape == Y[t].shape, f"'y_hat' wrong shape {y_hat.shape}"
                preds[t] = y_hat

            loss = self.loss(Y, preds)
            self.losses.append(loss)
            if epoch % print_loss == 0:
                samples = self.sample(h_last, inputs[0], num_samples)
                print(f"\nEpoch #{epoch}: {loss}")
                print(f"#### Sampled Text:\n {''.join(idx2char[idx] for idx in samples)} \n####")
                # Save model
                #self.save()

            # Move pointer by to next part of data
            pointer += self.seq_len
            # Update gradient
            self.backward(X, Y, preds)
            # Update parameters
            self.update_params(epoch)

    def forward(self, x, h_last):
        # hidden_size x 1
        a = self.b + matmul(self.W, h_last) + matmul(self.U, x)

        # hidden_size x 1
        h = torch.tanh(a)

        # alphabet_len x 1
        out = self.c + matmul(self.V, h)

        y_hat = torch.softmax(out, dim=0)
        return y_hat, h

    def backward(self, X, Y, preds):
        hs = self.hs
        dh_next = torch.zeros_like(hs[0])
        for t in reversed(range(self.seq_len)):
            y_hat = preds[t]
            y = Y[t]
            x = X[t]

            # alphabet_len x 1
            dy = y_hat - y
            # hidden_size x 1
            dh = matmul(self.V.t(), dy) + dh_next
            # hidden_size x 1
            da = (1 - hs[t]*hs[t]) * dh
            # hidden_size x 1
            dh_next = matmul(self.W.t(), da)

            dV = matmul(dy, hs[t].t())
            dW = matmul(da, hs[t-1].t())
            dU = matmul(da, x.t())
            # increment gradients:
            self.b.grad += da
            self.c.grad += dy

            self.V.grad += dV
            self.W.grad += dW
            self.U.grad += dU
        # prevent exploding gradient
        self.b.grad.clip_(-5, 5)
        self.c.grad.clip_(-5, 5)
        self.V.grad.clip_(-5, 5)
        self.W.grad.clip_(-5, 5)
        self.U.grad.clip_(-5, 5)

    def update_params(self, epoch):
        # Perform AdaGrad update
        self.memb += self.b.grad * self.b.grad
        self.memc += self.c.grad * self.c.grad
        self.memV += self.V.grad * self.V.grad
        self.memW += self.W.grad * self.W.grad
        self.memU += self.U.grad * self.U.grad

        self.b += -(self.eta/self.batch_size) * self.b.grad / torch.sqrt(self.memb + 1e-8)
        self.c += -(self.eta/self.batch_size) * self.c.grad / torch.sqrt(self.memc + 1e-8)
        self.V += -(self.eta/self.batch_size) * self.V.grad / torch.sqrt(self.memV + 1e-8)
        self.W += -(self.eta/self.batch_size) * self.W.grad / torch.sqrt(self.memW + 1e-8)
        self.U += -(self.eta/self.batch_size) * self.U.grad / torch.sqrt(self.memU + 1e-8)

    def zero_gradient(self):
        self.b.grad.zero_()
        self.c.grad.zero_()
        self.V.grad.zero_()
        self.W.grad.zero_()
        self.U.grad.zero_()
  
    def clear_state(self):
        self.hs = {}

    def predict(self, char: str, num_samples):
        idx2char = self.idx2char
        char2idx = self.char2idx
        idxs = self.sample(
            h=torch.zeros_like(self.hs[-1]), 
            seed_idx=char2idx[char],
            num_samples=num_samples
        )
        return "".join(idx2char[idx] for idx in idxs)

    def sample(self,
               h: torch.Tensor,
               seed_idx: torch.Tensor,
               num_samples: int):
        x = torch.zeros(self.alphabet_len, 1, device=self.device)
        x[seed_idx] = 1.0
        idxs = []
        for t in range(num_samples):
            probs, h = self.forward(x, h)
            sampled_index = probs.ravel().multinomial(num_samples=1)
            x.zero_()
            x[sampled_index] = 1.0
            idxs.append(sampled_index.item())
        return idxs

    def loss(self, Y, preds):
        """Cross-entropy loss for a sequence"""
        loss = 0
        for t in range(self.seq_len):
            # softmax probs
            y_hat = preds[t]
            # one-hot vector
            y = Y[t]
            # make y_hat numerically stable for log
            y_hat = y_hat * 0.99999 + 1e-10
            loss += (y * torch.log(y_hat)).sum().item()
        return -(loss/(self.seq_len-1))

    def save(self, dir_name="model_params"):
        torch.save(self.b, f"{dir_name}/b.pt")
        torch.save(self.c, f"{dir_name}/c.pt")
        torch.save(self.V, f"{dir_name}/V.pt")
        torch.save(self.W, f"{dir_name}/W.pt")
        torch.save(self.U, f"{dir_name}/U.pt")
 
    def load_params(self, dir_name: str):
        try:
            self.b = torch.load(f"{dir_name}/b.pt")
            self.b.grad = torch.zeros_like(self.b)
            self.c = torch.load(f"{dir_name}/c.pt")
            self.c.grad = torch.zeros_like(self.c)
            self.V = torch.load(f"{dir_name}/V.pt")
            self.V.grad = torch.zeros_like(self.V)
            self.W = torch.load(f"{dir_name}/W.pt")
            self.W.grad = torch.zeros_like(self.W)
            self.U = torch.load(f"{dir_name}/U.pt")
            self.U.grad = torch.zeros_like(self.U)
        except FileNotFoundError:
            print("\nCan't find parameters. Starting anew..")
