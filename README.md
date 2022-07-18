# Build2Understand
Build machine learning algorithms in order to learn how they work and create cool results.

## Recurrent Neural Networks
Implemented pure RNN (see [rnn.py](rnn.py)) which works okay. Next iteration will hopefully include GRU/LSTM and SGD for better optimization.
Additional Dense layers for better feature learning is also on the ToDo list. The goal is to support [this tutorial](https://keras.io/examples/generative/lstm_character_level_text_generation/) without using anything out of PyTorch's torch.nn module (e.g. only use torch for fast, GPU capable tensors). Code is written as if PyTorch were NumPy. No autograd is used.

Also want to train an RNN (LSTM or GRU) on ByteDances [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) dataset.

See [language_model.py](language_model.py) for an example of how to use the `SimpleRNN` in the `rnn` module.
See [binary_sequence.py](binary_sequence.py) for an even simpler example (used for testing).
