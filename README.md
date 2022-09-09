# Build2Understand
Build machine learning algorithms in order to learn how they work and create cool results.

## Recurrent Neural Networks
- Implemented pure RNN (see [rnn.py](rnn.py)) which works okay.
- Started implementing LSTM BPTT (backpropagation-through-time). See [lstm.py](lstm.py) for more.

The goal is still to support [this tutorial](https://keras.io/examples/generative/lstm_character_level_text_generation/) without using anything out of PyTorch's torch.nn module (e.g. only use torch for fast, GPU capable tensors). Code is written as if PyTorch were NumPy. No autograd is used.

Also want to train an RNN (LSTM or GRU) on ByteDances [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) dataset.

See [language_model.py](language_model.py) for an example of how to use the `SimpleRNN` in the `rnn` module.
See [binary_sequence.py](binary_sequence.py) for an even simpler example (used for testing).

See [language_model_LSTM.py](language_model_LSTM.py) and [binseq_LSTM.py](binseq_LSTM.py) for LSTM equivalent.
The LSTM doesn't function well yet. The language model seems to skew toward learning the endings more (e.g. "out originate out of its opposite?out of its opposite?" as a result, instead of "how could anything"). Will fix soon. At least the loss is being minimized..

### Literature
- [This paper](https://arxiv.org/abs/1503.04069) might shed some light on the troubles I'm having.
