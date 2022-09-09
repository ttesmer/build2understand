import torch
from torch import matmul, cat, sigmoid, tanh
from math import sqrt
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow

def sigmoid_prime(x):
    return x*(1-x)

class LSTM(object):
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
        Long Short-Term Memory Recurrent Neural Network
        """
        self.device = device
        self.epochs = epochs
        self.seq_len = sequence_len
        self.hidden_size = hidden_size
        self.alphabet_len = alphabet_len
        self.batch_size = batch_size
        self.eta = lr

        # Xavier weight initialization
        # (https://cs230.stanford.edu/section/4/)
        # don't know if this is right..

        # forget w&b
        self.Wf = torch.normal(
                torch.zeros(hidden_size, alphabet_len+hidden_size, device=device),
                std=1/(alphabet_len+hidden_size)
                )
        self.bf = torch.zeros(hidden_size, 1, device=device)
        # input w&b
        self.Wi = torch.normal(
                torch.zeros(hidden_size, alphabet_len+hidden_size, device=device),
                std=1/(alphabet_len+hidden_size)
                )
        self.bi = torch.zeros(hidden_size, 1, device=device)
        # activation w&b
        self.Wc = torch.normal(
                torch.zeros(hidden_size, alphabet_len+hidden_size, device=device),
                std=1/(alphabet_len+hidden_size)
                )
        self.bc = torch.zeros(hidden_size, 1, device=device)
        # output w&b
        self.Wo = torch.normal(
                torch.zeros(hidden_size, alphabet_len+hidden_size, device=device),
                std=1/(alphabet_len+hidden_size)
                )
        self.bo = torch.zeros(hidden_size, 1, device=device)
        # y_hat w&b
        self.Wy = torch.normal(
                torch.zeros(alphabet_len, hidden_size, device=device),
                std=1/hidden_size
                )
        self.by = torch.zeros(alphabet_len, 1, device=device)

        # gradients
        self.Wf.grad = torch.zeros_like(self.Wf)
        self.bf.grad = torch.zeros_like(self.bf)
        self.Wi.grad = torch.zeros_like(self.Wf)
        self.bi.grad = torch.zeros_like(self.bf)
        self.Wc.grad = torch.zeros_like(self.Wf)
        self.bc.grad = torch.zeros_like(self.bf)
        self.Wo.grad = torch.zeros_like(self.Wf)
        self.bo.grad = torch.zeros_like(self.bf)
        self.Wy.grad = torch.zeros_like(self.Wy)
        self.by.grad = torch.zeros_like(self.by)

        self.hs = {}
        self.Cs = {}
        self.C_hats = {}
        self.fs = {}
        self.is_ = {}
        self.os = {}
        self.losses = []

    def fit(self, data, num_samples, print_loss=1):
        h_last = torch.zeros(self.hidden_size, 1, device=self.device)
        C_last = torch.zeros_like(h_last)
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
                self.Cs[-1] = h_last

            # Get next part of sequence & its labels
            inputs = [char2idx[char] for char in data[pointer:pointer+self.seq_len]]
            targets = [char2idx[char] for char in data[pointer+1:pointer+self.seq_len+1]]
            assert len(inputs) == len(targets), f"len inputs: {len(inputs)} doesn't match len of targets: {len(targets)}"
            assert len(inputs) == self.seq_len, f"len inputs: {len(inputs)} doesn't match seq_len: {self.seq_len}"


            X = torch.zeros((len(inputs), self.alphabet_len, 1), device=self.device)
            Y = torch.zeros_like(X)
            preds = torch.zeros_like(X)
            self.hs[-1] = h_last
            self.Cs[-1] = C_last
            for t in range(len(inputs)):
                # Change index of input and target characters
                X[t][inputs[t]] = 1.0
                Y[t][targets[t]] = 1.0

                # Make sure the shape is a matrix
                assert X[t].shape == torch.Size([self.alphabet_len, 1]), f"X[t] wrong shape: {X[t].shape}"

                # Get softmax probcbilities and last state
                y_hat, h_last, C_last, C_hat, f, i, o = self.forward(X[t], h_last, C_last)
                # Add hidden state to correct time index
                self.C_hats[t] = C_hat
                self.hs[t] = h_last
                self.Cs[t] = C_last
                self.fs[t] = f
                self.is_[t] = i
                self.os[t] = o
                # Make sure the shape is a matrix
                assert y_hat.shape == Y[t].shape, f"'y_hat' wrong shape {y_hat.shape}"
                preds[t] = y_hat

            loss = self.loss(Y, preds)
            self.losses.append(loss)
            if epoch % print_loss == 0:
                samples = self.sample(h_last, C_last, inputs[0], num_samples)
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

    def forward(self, x_t, h_last, C_last):
        h_last_x = cat((h_last, x_t))
        # forget gate
        f_t = sigmoid((self.Wf @ h_last_x) + self.bf)
        # input gate
        i_t = sigmoid((self.Wi @ h_last_x) + self.bi)

        # Cell state
        C_hat = tanh((self.Wc @ h_last_x) + self.bc)
        C_t = f_t * C_last + i_t * C_hat

        # output gate
        o_t = sigmoid((self.Wo @ h_last_x) + self.bo)

        # hidden state
        h_t = o_t * tanh(C_t)

        y_hat = torch.softmax((self.Wy @ h_t) + self.by, dim=0)
        return y_hat, h_t, C_t, C_hat, f_t, i_t, o_t

    def backward(self, X, Y, preds):
        hs = self.hs
        Cs = self.Cs
        dh_next = torch.zeros_like(hs[0])
        dC_next = torch.zeros_like(Cs[0])
        for t in reversed(range(self.seq_len)):
            y_hat = preds[t]
            y = Y[t]
            x = X[t]
            h = hs[t]
            h_last = hs[t-1]
            C = Cs[t]
            C_last = Cs[t-1]
            C_hat = self.C_hats[t]
            f = self.fs[t]
            i = self.is_[t]
            o = self.os[t]
            tanhC_t = tanh(C)
            # (only calculate h_last_x once)
            h_last_x = cat((h_last, x))
            h_last_xT = cat((h_last, x)).t()

            # reverse mode differentiation
            # TODO: write an autograd
            dy = y_hat - y
            # hidden_size x 1
            dh = matmul(self.Wy.t(), dy) + dh_next
            dC = ((1 - tanhC_t.square()) * dh * o) + dC_next
            dC_next = f * dC
            dC_hat = (1 - (C_hat).square()) * (i * dC)

            # all are hidden_size x 1
            do = sigmoid_prime(o) * (tanhC_t * dh)
            di = sigmoid_prime(i) * (C_hat * dC)
            df = sigmoid_prime(f) * (C_last * dC)
            print("mean f value:", f.mean().item())
            print("mean i value:", i.mean().item())
            print("mean o value:", o.mean().item())
            print("mean C value:", C.mean().item())

            # print(df.sum().item(), di.sum().item(), do.sum().item())
            # print("\nf:", f[0][0].item())
            # print("dC: ", dC[0][0].item())
            # print("Components of dC:")
            # print("o: ", o[0][0].item())
            # print("tanhC_t: ", tanhC_t[0][0].item())
            # print("dh: ", dh[0][0].item())
            # print("Components of dh:")
            # print("Wy: ", self.Wy[0][0].item())
            # print("dh_next: ", dh_next[0][0].item())
            # print()

            dXf = matmul(self.Wf.t(), df)
            dXi = matmul(self.Wi.t(), di)
            dXc = matmul(self.Wc.t(), dC)
            dXo = matmul(self.Wo.t(), do)

            dX = dXf + dXi + dXc + dXo

            dh_next = dX[:self.hidden_size]

            # gradients
            dWy = matmul(dy, h.t())
            dby = dy
            dWo = matmul(do, h_last_xT)
            dbo = do
            dWc = matmul(dC_hat, h_last_xT)
            dbc = dC_hat
            dWi = matmul(di, h_last_xT)
            dbi = di
            dWf = matmul(df, h_last_xT)
            dbf = df

            # updates
            self.Wy.grad += dWy
            self.by.grad += dby
            self.Wo.grad += dWo
            self.bo.grad += dbo
            self.Wc.grad += dWc
            self.bc.grad += dbc
            self.Wi.grad += dWi
            self.bi.grad += dbi
            self.Wf.grad += dWf
            self.bf.grad += dbf

        # # prevent exploding gradient
        # # per time step or after sequence backprop?
        # self.Wf.grad.clip_(-5, 5)
        # self.bf.grad.clip_(-5, 5)
        # self.Wi.grad.clip_(-5, 5)
        # self.bi.grad.clip_(-5, 5)
        # self.Wc.grad.clip_(-5, 5)
        # self.bc.grad.clip_(-5, 5)
        # self.Wo.grad.clip_(-5, 5)
        # self.bo.grad.clip_(-5, 5)
        # self.Wy.grad.clip_(-5, 5)
        # self.by.grad.clip_(-5, 5)
    
    def update_params(self, epoch):
        print(self.Wf.grad)
        self.Wf += -self.eta * self.Wf.grad
        self.bf += -self.eta * self.bf.grad
        self.Wi += -self.eta * self.Wi.grad
        self.bi += -self.eta * self.bi.grad
        self.Wc += -self.eta * self.Wc.grad
        self.bc += -self.eta * self.bc.grad
        self.Wo += -self.eta * self.Wo.grad
        self.bo += -self.eta * self.bo.grad
        self.Wy += -self.eta * self.Wy.grad
        self.by += -self.eta * self.by.grad

    def zero_gradient(self):
        self.Wf.grad.zero_()
        self.Wi.grad.zero_()
        self.bf.grad.zero_()
        self.bi.grad.zero_()
        self.Wc.grad.zero_()
        self.bc.grad.zero_()
        self.Wo.grad.zero_()
        self.bo.grad.zero_()
        self.Wy.grad.zero_()
        self.by.grad.zero_()
  
    def clear_state(self):
        self.hs = {}
        self.Cs = {}
        self.C_hats = {}
        self.fs = {}
        self.is_ = {}
        self.os = {}

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
               C: torch.Tensor,
               seed_idx: torch.Tensor,
               num_samples: int):
        x = torch.zeros(self.alphabet_len, 1, device=self.device)
        x[seed_idx] = 1.0
        idxs = [seed_idx]
        for t in range(num_samples):
            probs, h, C, *_ = self.forward(x, h, C)
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
