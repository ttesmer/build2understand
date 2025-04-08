import torch
from torch import matmul, cat, tanh, sigmoid
from collections import OrderedDict
from typing import List, Tuple
#torch.manual_seed(25)

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
                 beta: float,
                 batch_size=1):
        super().__init__()
        """
        Long Short-Term Memory Block
        (with output layer)
        """
        self.device = device
        self.epochs = epochs
        self.seq_len = sequence_len
        self.hidden_size = hidden_size
        self.alphabet_len = alphabet_len
        self.batch_size = batch_size
        self.eta = lr
        self.beta = beta

        # Xavier weight initialization
        # (https://cs230.stanford.edu/section/4/)
        # don't know if this is right..

        # forget w&b
        self.Wf = torch.randn((hidden_size, alphabet_len), device=device) * 1e-2
        self.Rf = torch.randn((hidden_size, hidden_size), device=device) * 1e-2
        self.bf = torch.randn(hidden_size, 1, device=device) * 1e-2
        # input w&b
        self.Wi = torch.randn((hidden_size, alphabet_len), device=device) * 1e-2
        self.Ri = torch.randn((hidden_size, hidden_size), device=device) * 1e-2
        self.bi = torch.randn(hidden_size, 1, device=device) * 1e-2
        # activation w&b
        self.Wc = torch.randn((hidden_size, alphabet_len), device=device) * 1e-2
        self.Rc = torch.randn((hidden_size, hidden_size), device=device) * 1e-2
        self.bc = torch.randn(hidden_size, 1, device=device) * 1e-2
        # output w&b
        self.Wo = torch.randn((hidden_size, alphabet_len), device=device) * 1e-2
        self.Ro = torch.randn((hidden_size, hidden_size), device=device) * 1e-2
        self.bo = torch.randn(hidden_size, 1, device=device) * 1e-2
        # y_hat w&b
        self.Wy = torch.randn((alphabet_len, hidden_size), device=device) * 1e-2
        self.by = torch.randn(alphabet_len, 1, device=device) * 1e-2

        # gradients
        self.Wf.grad = torch.zeros_like(self.Wf)
        self.Rf.grad = torch.zeros_like(self.Rf)
        self.bf.grad = torch.zeros_like(self.bf)
        self.Wi.grad = torch.zeros_like(self.Wf)
        self.Ri.grad = torch.zeros_like(self.Ri)
        self.bi.grad = torch.zeros_like(self.bf)
        self.Wc.grad = torch.zeros_like(self.Wf)
        self.Rc.grad = torch.zeros_like(self.Rc)
        self.bc.grad = torch.zeros_like(self.bf)
        self.Wo.grad = torch.zeros_like(self.Wf)
        self.Ro.grad = torch.zeros_like(self.Ro)
        self.bo.grad = torch.zeros_like(self.bf)
        self.Wy.grad = torch.zeros_like(self.Wy)
        self.by.grad = torch.zeros_like(self.by)

        self.hs = {}
        self.C = {}
        self.C_hat = {}
        self.f = {}
        self.i = {}
        self.o = {}
        self.losses = []

    def parameters(self):
        """
        Returns [params]
        """
        params = [
                self.Wf,
                self.Rf,
                self.bf,
                self.Wi,
                self.Ri,
                self.bi,
                self.Wc,
                self.Rc,
                self.bc,
                self.Wo,
                self.Ro,
                self.bo,
                self.Wy,
                self.by
                ]
        return params


    def fit(self,
            X,
            Y,
            num_iters,
            weight_decay,
            num_samples=10,
            print_loss=1):
        # Do forward & backward pass `num_iters` times.
        # Update parameters each time both passes are completed.
        self.weight_decay = weight_decay
        for epoch in range(1, num_iters+1):
            self.clear_state()
            self.init_state_zero()
            self.zero_grad()
            h_last = self.hs[-1]
            state = (h_last, h_last)

            probs, caches, states = self.forward_sequence(X, state)
            loss = self.backward_sequence(X, Y, probs, caches, states)

            self.update_params()

            if epoch % print_loss == 0:
                seed = X[-1]
                print(f"\nEpoch #{epoch}: {loss}")
                print('...Generating with seed char: "' + self.idx2char[seed.argmax().item()] + '"')
                print("#"*4, "Generated text after last char of seed:")
                for diversity in [0.2, 0.5, 1.0, 1.2]:
                    print(f"..Diversity: {diversity}")
                    samples = self.sample_sequence(
                            seed=seed,
                            state=self.hs[-1],
                            num_samples=5,
                            temperature=diversity)
                    print(''.join(self.idx2char[idx] for idx in samples))
                print("#"*4)


    def forward_sequence(self, X, state, train=True):
        probs = {}
        caches = {}
        states = {-1: state}
        for t in range(X.shape[0]):
            # last state
            x = X[t]
            h_last, c_last = state

            # forward pass
            f = sigmoid(self.Wf  @ x + self.Rf @ h_last + self.bf)
            i = sigmoid(self.Wi  @ x + self.Ri @ h_last + self.bi)
            c_hat = tanh(self.Wc @ x + self.Rc @ h_last + self.bc)
            c = f * c_last + i * c_hat
            o = sigmoid(self.Wo  @ x + self.Ro @ h_last + self.bo)
            h = o * tanh(c)

            # output layer
            y_hat = torch.softmax((self.Wy @ h) + self.by, dim=0)

            # new state
            state = (h, c)

            probs[t] = y_hat
            states[t] = state
            caches[t] = (f, i, c_hat, c, o, h)
            # print("h_last", h_last)
            # print("f", f)
            # print("i", i)
            # print(f"t: {t}")
            # print("o", o)
            # print("C_hat", c_hat)
            # print("C", tanh(c))
            # print("y_hat", y_hat)
            # print()
        if not train: 
            return probs, state
        return probs, caches, states

    def backward_sequence(self, X, Y, probs, caches, states):
        loss = 0
        dh_next, dc_next = states[-1]
        for t in reversed(range(X.shape[0])):
            f, i, c_hat, c, o, h = caches[t]
            h_last, c_last = states[t-1]
            x = X[t]

            dy = probs[t] - Y[t]
            dh = matmul(self.Wy.t(), dy) + dh_next
            dc = (1. - c.square()) * o * dh + dc_next
            dc_hat = (1. - c_hat.square()) * i * dc

            do = sigmoid_prime(o) * tanh(c) * dh
            di = sigmoid_prime(i) * c_hat * dc
            df = sigmoid_prime(f) * c_last * dc

            dhf = matmul(self.Rf.t(), df)
            dhi = matmul(self.Ri.t(), di)
            dhc = matmul(self.Rc.t(), dc)
            dho = matmul(self.Ro.t(), do)
            dh_next = dhf + dhi + dhc + dho
            dc_next = f * dc

            self.Wy.grad += matmul(dy, h.t())
            self.by.grad += dy
            self.Wo.grad += matmul(do, x.t())
            self.Ro.grad += matmul(do, h_last.t())
            self.bo.grad += do
            self.Wc.grad += matmul(dc_hat, x.t())
            self.Rc.grad += matmul(dc_hat, h_last.t())
            self.bc.grad += dc_hat
            self.Wi.grad += matmul(di, x.t())
            self.Ri.grad += matmul(di, h_last.t())
            self.bi.grad += di
            self.Wf.grad += matmul(df, x.t())
            self.Rf.grad += matmul(df, h_last.t())
            self.bf.grad += df
            loss += self.cross_entropy(Y[t], probs[t])
        return -loss/X.shape[0]

    def sample_sequence(self,
               seed: torch.Tensor,
               state: Tuple,
               num_samples: int,
               temperature=1.0):
        idxs = []
        state = (torch.zeros_like(self.hs[-1]), torch.zeros_like(self.C[-1]))
        for t in range(num_samples):
            seed = seed.to(torch.float32)
            probs, state = self.forward(seed, state, cache_vars=False)
            probs = probs.to(torch.float32)
            probs = probs.log() / temperature
            exp_probs = probs.exp()
            probs = exp_probs / exp_probs.sum()
            sampled_index = probs.ravel().multinomial(num_samples=1)
            seed.zero_()
            seed[sampled_index] = 1.0
            idxs.append(sampled_index.item())
        return idxs

    def update_params(self):
        for param in self.parameters():
            # L2 Regularization/Ridge Regression/Weight Decay
            # from here: https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
            # param += (1 - self.eta * self.weight_decay) - self.eta * param.grad
            param += -self.eta * param.grad

    def zero_grad(self):
        for param in self.parameters():
            param.grad.zero_()
  
    def clear_state(self):
        self.hs = {}
        self.C = {}
        self.C_hat = {}
        self.f = {}
        self.i = {}
        self.o = {}

    def init_state_zero(self):
        self.hs[-1] = torch.zeros(self.hidden_size, 1, device=self.device)
        self.C[-1] = torch.zeros(self.hidden_size, 1, device=self.device)

    def backward_step_single(self, x, y, y_hat, dstate, t):
        hs = self.hs
        Cs = self.C
        dh_next, dC_next = dstate

        h = hs[t]
        h_last = hs[t-1]
        C = Cs[t]
        C_last = Cs[t-1]
        C_hat = self.C_hat[t]
        f = self.f[t]
        i = self.i[t]
        o = self.o[t]
        tanhC_t = tanh(C)
        # (only calculate h_last_x once)
        h_last_x = cat((h_last, x))
        h_last_xT = h_last_x.t()

        dy = y_hat - y
        dh = matmul(self.Wy.t(), dy) + dh_next
        dC = ((1. - tanhC_t.square()) * dh * o) + dC_next
        dC_hat = (1. - (C_hat).square()) * (i * dC)

        # all are hidden_size x 1
        do = sigmoid_prime(o) * (tanhC_t * dh)
        di = sigmoid_prime(i) * (C_hat * dC)
        df = sigmoid_prime(f) * (C_last * dC)

        dXf = matmul(self.Wf.t(), df)
        dXi = matmul(self.Wi.t(), di)
        dXc = matmul(self.Wc.t(), dC)
        dXo = matmul(self.Wo.t(), do)

        dX = dXf + dXi + dXc + dXo

        dh_next = dX[:self.hidden_size, :]
        dC_next = f * dC

        # increment gradients
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

        grads = (dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby)
        dstate = (dh_next, dC_next)
        return grads, dstate

    def forward(self, x_t, state, t=1, cache_vars=True):
        h_last, c_last = state

        f_t = sigmoid(self.Wf @ x_t + self.Rf @ h_last + self.bf)
        i_t = sigmoid(self.Wi @ x_t + self.Ri @ h_last + self.bi)
        C_hat = torch.tanh(self.Wc @ x_t + self.Rc @ h_last + self.bc)
        C_t = f_t * c_last + i_t * C_hat
        o_t = sigmoid(self.Wo @ x_t + self.Ro @ h_last + self.bo)
        h_t = o_t * tanh(C_t)

        state = (h_t, C_t)
        cache = (f_t, i_t, C_t, o_t)
        if cache_vars:
            self.f[t] = f_t
            self.i[t] = i_t
            self.C[t] = C_t
            self.o[t] = o_t
            self.hs[t] = h_t
            self.C_hat[t] = C_hat

        y_hat = torch.softmax((self.Wy @ h_t) + self.by, dim=0)
        if cache_vars: 
            return y_hat, state, cache
        else:
            return y_hat, state

    def predict(self, char: str, num_samples):
        idx2char = self.idx2char
        char2idx = self.char2idx
        idxs = self.sample(
            h=torch.zeros_like(self.hs[-1]), 
            seed_idx=char2idx[char],
            num_samples=num_samples
        )
        return "".join(idx2char[idx] for idx in idxs)

    def cross_entropy(self, y, y_hat):
        """Cross-entropy loss for a single prediction"""
        y_hat *= 0.99999 + 1e-10
        loss = (y * torch.log(y_hat)).sum().item()
        return loss


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
