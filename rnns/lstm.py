import torch
from torch import matmul, cat, tanh, sigmoid
from collections import OrderedDict
from typing import List, Tuple
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
torch.manual_seed(25)

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
        self.Wf = torch.randn((hidden_size, alphabet_len+hidden_size), device=device) * 1e-2
        self.bf = torch.randn(hidden_size, 1, device=device) * 1e-2
        # input w&b
        self.Wi = torch.randn((hidden_size, alphabet_len+hidden_size), device=device) * 1e-2
        self.bi = torch.randn(hidden_size, 1, device=device) * 1e-2
        # activation w&b
        self.Wc = torch.randn((hidden_size, alphabet_len+hidden_size), device=device) * 1e-2
        self.bc = torch.randn(hidden_size, 1, device=device) * 1e-2
        # output w&b
        self.Wo = torch.randn((hidden_size, alphabet_len+hidden_size), device=device) * 1e-2
        self.bo = torch.randn(hidden_size, 1, device=device) * 1e-2
        # y_hat w&b
        self.Wy = torch.randn((alphabet_len, hidden_size), device=device) * 1e-2
        self.by = torch.randn(alphabet_len, 1, device=device) * 1e-2

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

        self.memWf = torch.zeros_like(self.Wf)
        self.membf = torch.zeros_like(self.bf)
        self.memWi = torch.zeros_like(self.Wf)
        self.membi = torch.zeros_like(self.bf)
        self.memWc = torch.zeros_like(self.Wf)
        self.membc = torch.zeros_like(self.bf)
        self.memWo = torch.zeros_like(self.Wf)
        self.membo = torch.zeros_like(self.bf)
        self.memWy = torch.zeros_like(self.Wy)
        self.memby = torch.zeros_like(self.by)

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
                self.bf,
                self.Wi,
                self.bi,
                self.Wc,
                self.bc,
                self.Wo,
                self.bo,
                self.Wy,
                self.by
                ]
        mem_params = [
                self.memWf,
                self.membf,
                self.memWi,
                self.membi,
                self.memWc,
                self.membc,
                self.memWo,
                self.membo,
                self.memWy,
                self.memby
                ]
        return params, mem_params


    def fit_no_SGD(
            self,
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

            # if epoch % print_loss == 0:
            print(f"\nEpoch #{epoch}: {loss}")
            # print('...Generating with seed: "' + ''.join(self.idx2char[idx.item()] for idx in X.argmax(dim=1)) + '"')
            # print("#"*4, "Generated text after last char of seed:")
            # for diversity in [0.2, 0.5, 1.0, 1.2]:
            #     print(f"..Diversity: {diversity}")
            #     samples = self.sample_sequence(
            #             seed=X.argmax(dim=1)[-self.alphabet_len:],
            #             state=state,
            #             num_samples=self.seq_len,
            #             temperature=diversity)
            #     print(''.join(self.idx2char[idx] for idx in samples))
            # print("#"*4)

    def forward_sequence(self, X, state, train=True):
        probs = {}
        caches = {}
        states = {-1: state}
        for t in range(X.shape[0]):
            # last state
            h_last, c_last = state
            h_last_x = cat((h_last, X[t]))

            # forward pass
            f = sigmoid((self.Wf @ h_last_x) + self.bf)
            i = sigmoid((self.Wi @ h_last_x) + self.bi)
            c_hat = tanh((self.Wc @ h_last_x) + self.bc)
            c = f * c_last + i * c_hat
            o = sigmoid((self.Wo @ h_last_x) + self.bo)
            h = o * tanh(c)
            # print(f"t: {t}", f"h max:", max(h).item())

            # output layer
            y_hat = torch.softmax((self.Wy @ h) + self.by, dim=0)

            # new state
            state = (h, c)

            probs[t] = y_hat
            states[t] = state
            caches[t] = (f, i, c_hat, c, o, h)
            print("h_last_x", h_last_x)
            print("f", f)
            print("i", i)
            print(f"t: {t}")
            print("o", o)
            print("C_hat", c_hat)
            print("C", tanh(c))
            print("y_hat", y_hat)
            print()
        # exit()
        if not train: 
            return probs, state
        return probs, caches, states

    def backward_sequence(self, X, Y, probs, caches, states):
        loss = 0
        dh_next, dc_next = states[-1]
        for t in reversed(range(X.shape[0])):
            f, i, c_hat, c, o, h = caches[t]
            h_last, c_last = states[t-1]
            h_last_x = cat((h_last, X[t]))
            h_last_xT = h_last_x.t()

            dy = probs[t] - Y[t]
            dh = matmul(self.Wy.t(), dy) + dh_next
            dc = (1. - c.square()) * dh * o + dc_next
            dc_hat = (1. - c_hat.square()) * dc * i

            do = sigmoid_prime(o) * (tanh(c) * dh)
            di = sigmoid_prime(i) * (c_hat * dc)
            df = sigmoid_prime(f) * (c_last * dc)
            print(f"t: {t}")
            print("do:", do)
            # print(f"t: {t}", f"dh_next max:", max(dh_next).item())
            # print(f"t: {t}", f"do max:", max(do).item())
            # print(f"t: {t}", f"di max:", max(di).item())
            # print(f"t: {t}", f"df max:", max(df).item())
            # print(f"t: {t}", f"dc max:", max(dc).item())

            dXf = matmul(self.Wf.t(), df)
            dXi = matmul(self.Wi.t(), di)
            dXc = matmul(self.Wc.t(), dc)
            dXo = matmul(self.Wo.t(), do)
            dX = dXf + dXi + dXc + dXo
            dh_next = dX[:self.hidden_size, :]
            dc_next = f * dc

            self.Wy.grad += matmul(dy, h.t())
            self.by.grad += dy
            self.Wo.grad += matmul(do, h_last_xT)
            self.bo.grad += do
            self.Wc.grad += matmul(dc_hat, h_last_xT)
            self.bc.grad += dc_hat
            self.Wi.grad += matmul(di, h_last_xT)
            self.bi.grad += di
            self.Wf.grad += matmul(df, h_last_xT)
            self.bf.grad += df
            loss += self.cross_entropy(Y[t], probs[t])
        return -loss/X.shape[0]

    def sample_sequence(self,
               seed: torch.Tensor,
               state: Tuple,
               num_samples: int,
               temperature=1.0):
        idxs = [seed.argmax().item()]
        state = (torch.zeros_like(self.hs[-1]), torch.zeros_like(self.C[-1]))
        for t in range(num_samples):
            probs, state = self.forward(seed, state, cache_vars=False)
            probs = probs.to(torch.float64)
            probs = probs.log() / temperature
            exp_probs = probs.exp()
            probs = exp_probs / exp_probs.sum()
            sampled_index = probs.ravel().multinomial(num_samples=1)
            seed.zero_()
            seed[sampled_index] = 1.0
            idxs.append(sampled_index.item())
        return idxs

    def update_params(self):
        for param, mem_param in zip(*self.parameters()):
            mem_param = self.beta * mem_param + (1 - self.beta) * param.grad.square()
            # param += -(self.eta/torch.sqrt(mem_param + 1e-8)) * param.grad

            # L2 Regularization/Ridge Regression/Weight Decay
            # from here: https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
            param += (1 - self.eta * self.weight_decay) - self.eta * param.grad

    def zero_grad(self):
        for param, _ in zip(*self.parameters()):
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

    def fit_(self, data, num_samples, print_loss=1):
        h_last = torch.zeros(self.hidden_size, 1, device=self.device)
        state = (h_last, h_last)
        self.hs[-1] = state[0]
        self.C[-1] = state[1]
        probs = []
        caches = []
        pointer = 0
        loss = 0
        alphabet = sorted(list(set(data)))
        char2idx = {c: i for i,c in enumerate(alphabet)}
        idx2char = {i: c for i,c in enumerate(alphabet)}
        self.char2idx = char2idx
        self.idx2char = idx2char
        for epoch in range(self.epochs):
            mini_batches = []
            for mb_idx in range(0, self.seq_len, self.batch_size):
                # print(mb_idx, mb_idx+self.batch_size)
                mbx = data[mb_idx:mb_idx+self.batch_size]
                mby = data[mb_idx+1:mb_idx+self.batch_size+1]
                mini_batches.append((mbx, mby))
            # print(f"\nMini-batches: \n{mini_batches}")

            self.clear_state()
            self.init_state_zero()
            dWf = torch.zeros_like(self.Wf)
            dbf = torch.zeros_like(self.bf)
            dWi = torch.zeros_like(self.Wf)
            dbi = torch.zeros_like(self.bf)
            dWc = torch.zeros_like(self.Wf)
            dbc = torch.zeros_like(self.bf)
            dWo = torch.zeros_like(self.Wf)
            dbo = torch.zeros_like(self.bf)
            dWy = torch.zeros_like(self.Wy)
            dby = torch.zeros_like(self.by)
            grads = dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby
            for x, y in mini_batches:
                seq_len = self.batch_size
                # Reset gradient & hidden state
                self.zero_gradient()
                # Reset pointer to not go out of index bounds
                if pointer+seq_len+1 >= len(x):
                    pointer = 0
                    h_last = torch.zeros(self.hidden_size, 1, device=self.device)
                    state = (h_last, h_last)

                # Gx, y et next part of sequence & its labels
                inputs = [char2idx[char] for char in x]
                targets = [char2idx[char] for char in x]
                assert len(inputs) == len(targets), f"len inputs: {len(inputs)} doesn't match len of targets: {len(targets)}"
                assert len(inputs) == seq_len, f"len inputs: {len(inputs)} doesn't match seq_len: {seq_len}"

                X = torch.zeros((len(inputs), self.alphabet_len, 1), device=self.device)
                Y = torch.zeros_like(X)
                preds = torch.zeros_like(X)


                for t in range(self.batch_size):
                    # Change index of input and target characters
                    X[t][inputs[t]] = 1.0
                    Y[t][targets[t]] = 1.0

                    # Make sure the shape is a matrix
                    assert X[t].shape == torch.Size([self.alphabet_len, 1]), f"X[t] wrong shape: {X[t].shape}"

                    # Get softmax probcbilities and last state
                    y_hat, state, cache = self.forward(X[t], state, t)

                    # Make sure the shape is a matrix
                    assert y_hat.shape == Y[t].shape, f"'y_hat' wrong shape {y_hat.shape}"
                    preds[t] = y_hat
                    # print(y_hat)

                    probs.append(y_hat)
                    caches.append(cache)
                pointer += self.seq_len
                grads = self.backward_step(X, Y, preds, grads)
            
            # self.backward(X, Y, preds)
            self.update_mini_batch(grads)

            # loss = self.loss(Y, preds)
            # self.losses.append(loss)

            if epoch % print_loss == 0:
                loss = 0
                print(f"\nEpoch #{epoch}: {loss}")
                for diversity in [0.2, 0.5, 1.0, 1.2]:
                    print(f"\n..Diversity: {diversity}\r")
                    samples = self.sample((self.hs[-1], self.C[-1]), inputs[0], num_samples, diversity)
                    print(f"#### Sampled Text:\n {''.join(idx2char[idx] for idx in samples)} \n####")



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
        h_last_x = cat((h_last, x_t))

        # forget gate
        f_t = sigmoid((self.Wf @ h_last_x) + self.bf)
        # f_t = 1.0
        # input gate
        i_t = sigmoid((self.Wi @ h_last_x) + self.bi)

        # Cell state
        C_hat = torch.tanh((self.Wc @ h_last_x) + self.bc)
        C_t = f_t * c_last + i_t * C_hat

        # output gate
        o_t = sigmoid((self.Wo @ h_last_x) + self.bo)

        # hidden state
        h_t = o_t * tanh(C_t)

        state = (h_t, C_t)
        # don't think we need the cache;
        # current implementation seems right
        cache = (f_t, i_t, C_t, o_t)
        if cache_vars:
            self.f[t] = f_t
            self.i[t] = i_t
            self.C[t] = C_t
            self.o[t] = o_t
            self.hs[t] = h_t
            self.C_hat[t] = C_hat

        y_hat = torch.softmax((self.Wy @ h_t) + self.by, dim=0)
        # print("h_last_x", h_last_x)
        # print("f", f_t)
        # print("i", i_t)
        # print("o", o_t)
        # print("C_hat", C_hat)
        # print("C", tanh(C_t))
        # print("y_hat", y_hat)
        # print()
        # if f_t.sum().item() == f_t.shape[0]:
        #     exit()

        if cache_vars: 
            return y_hat, state, cache
        else:
            return y_hat, state

    def backward_step(self, X, Y, preds, grads):
        dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby = grads
        self.zero_gradient()
        hs = self.hs
        Cs = self.C
        dh_next = torch.zeros_like(hs[0])
        dC_next = torch.zeros_like(Cs[0])
        for t in reversed(range(self.batch_size)):
            y_hat = preds[t]
            y = Y[t]
            x = X[t]
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
            dC_hat = (1. - tanh(C_hat).square()) * (i * dC)

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
            dWy += matmul(dy, h.t())
            dby += dy
            dWo += matmul(do, h_last_xT)
            dbo += do
            dWc += matmul(dC_hat, h_last_xT)
            dbc += dC_hat
            dWi += matmul(di, h_last_xT)
            dbi += di
            dWf += matmul(df, h_last_xT)
            dbf += df
        grads = dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby
        return grads

    def update_mini_batch(self, grads):
        dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby = grads

        self.memWf = self.beta * self.memWf + (1 - self.beta) * dWf.square()
        self.membf = self.beta * self.membf + (1 - self.beta) * dbf.square()
        self.memWi = self.beta * self.memWi + (1 - self.beta) * dWi.square()
        self.membi = self.beta * self.membi + (1 - self.beta) * dbi.square()
        self.memWc = self.beta * self.memWc + (1 - self.beta) * dWc.square()
        self.membc = self.beta * self.membc + (1 - self.beta) * dbc.square()
        self.memWo = self.beta * self.memWo + (1 - self.beta) * dWo.square()
        self.membo = self.beta * self.membo + (1 - self.beta) * dbo.square()
        self.memWy = self.beta * self.memWy + (1 - self.beta) * dWy.square()
        self.memby = self.beta * self.memby + (1 - self.beta) * dby.square()

        self.Wf -= self.eta / (torch.sqrt(self.memWf) + 1e-8) * dWf
        self.bf -= self.eta / (torch.sqrt(self.membf) + 1e-8) * dbf
        self.Wi -= self.eta / (torch.sqrt(self.memWi) + 1e-8) * dWi
        self.bi -= self.eta / (torch.sqrt(self.membi) + 1e-8) * dbi
        self.Wc -= self.eta / (torch.sqrt(self.memWc) + 1e-8) * dWc
        self.bc -= self.eta / (torch.sqrt(self.membc) + 1e-8) * dbc
        self.Wo -= self.eta / (torch.sqrt(self.memWo) + 1e-8) * dWo
        self.bo -= self.eta / (torch.sqrt(self.membo) + 1e-8) * dbo
        self.Wy -= self.eta / (torch.sqrt(self.memWy) + 1e-8) * dWy
        self.by -= self.eta / (torch.sqrt(self.memby) + 1e-8) * dby

    def backward(self, X, Y, preds):
        dWf = torch.zeros_like(self.Wf)
        dbf = torch.zeros_like(self.bf)
        dWi = torch.zeros_like(self.Wf)
        dbi = torch.zeros_like(self.bf)
        dWc = torch.zeros_like(self.Wf)
        dbc = torch.zeros_like(self.bf)
        dWo = torch.zeros_like(self.Wf)
        dbo = torch.zeros_like(self.bf)
        dWy = torch.zeros_like(self.Wy)
        dby = torch.zeros_like(self.by)
        self.zero_gradient()
        hs = self.hs
        Cs = self.C
        dh_next = torch.zeros_like(hs[0])
        dC_next = torch.zeros_like(Cs[0])
        for t in reversed(range(self.batch_size)):
            y_hat = preds[t]
            y = Y[t]
            x = X[t]
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

            # gradients
            dWy += matmul(dy, h.t())
            dby += dy
            dWo += matmul(do, h_last_xT)
            dbo += do
            dWc += matmul(dC_hat, h_last_xT)
            dbc += dC_hat
            dWi += matmul(di, h_last_xT)
            dbi += di
            dWf += matmul(df, h_last_xT)
            dbf += df

        self.memWf = self.beta * self.memWf + (1 - self.beta) * dWf.square()
        self.membf = self.beta * self.membf + (1 - self.beta) * dbf.square()
        self.memWi = self.beta * self.memWi + (1 - self.beta) * dWi.square()
        self.membi = self.beta * self.membi + (1 - self.beta) * dbi.square()
        self.memWc = self.beta * self.memWc + (1 - self.beta) * dWc.square()
        self.membc = self.beta * self.membc + (1 - self.beta) * dbc.square()
        self.memWo = self.beta * self.memWo + (1 - self.beta) * dWo.square()
        self.membo = self.beta * self.membo + (1 - self.beta) * dbo.square()
        self.memWy = self.beta * self.memWy + (1 - self.beta) * dWy.square()
        self.memby = self.beta * self.memby + (1 - self.beta) * dby.square()

        self.Wf -= self.eta / (torch.sqrt(self.memWf) + 1e-8) * dWf
        self.bf -= self.eta / (torch.sqrt(self.membf) + 1e-8) * dbf
        self.Wi -= self.eta / (torch.sqrt(self.memWi) + 1e-8) * dWi
        self.bi -= self.eta / (torch.sqrt(self.membi) + 1e-8) * dbi
        self.Wc -= self.eta / (torch.sqrt(self.memWc) + 1e-8) * dWc
        self.bc -= self.eta / (torch.sqrt(self.membc) + 1e-8) * dbc
        self.Wo -= self.eta / (torch.sqrt(self.memWo) + 1e-8) * dWo
        self.bo -= self.eta / (torch.sqrt(self.membo) + 1e-8) * dbo
        self.Wy -= self.eta / (torch.sqrt(self.memWy) + 1e-8) * dWy
        self.by -= self.eta / (torch.sqrt(self.memby) + 1e-8) * dby

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

    def clip_gradient_norm(self, grads, max_norm=0.25):
        """
        Clips gradients to have a maximum norm of `max_norm`.
        This is to prevent the exploding gradient problem.
        """
        max_norm = float(max_norm)
        total_norm = 0

        # Calculate the L2 norm squared for each gradient and add them to the total norm
        for grad in grads:
            grad_norm = grad.square().sum()
            total_norm += grad_norm

        total_norm.sqrt_()

        # Clipping coefficient
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for grad in grads:
                grad *= clip_coef
        return grads
