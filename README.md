# Build2Understand
Build machine learning algorithms in order to learn how they work and create cool results.

## Transformers
- Implemented GPT-2 in PyTorch ([transformer/GPT-2 inference.ipynb](transformer/4%20full%20GPT-2%20inference.ipynb))
- Notably, I didn't follow the code implementation, but rather the config file and some blog posts. Because of this, I had to manually assign each weight matrix, transpose some (since OpenAI used a Conv1D instead of matmul in some places, so we have to transpose to get nn.Linear equivalent), and split the attention matrices. This is because I decided to implement each AttentionHead itself, not in one matrix (which is more efficient, but not very readable or good for didactic purposes. The `attn.c_attn.weight` from OpenAI is `768x2304`, which is actually all Q, K, V matrices together for that AttentionBlock: `2304=768*3` (the three are the Q, K, V of the block). So that's actually the Q, K, V. Each with dimensions `768x768`. But even that's not what we mean mathematically on a high level.. *Those* are the 12 (`12*64=768`) AttentionHeads in dim=1. So, my implementation is the first (that I've seen), that is arguably slower (though not noticeably on my machine), but IMO *much* more didactically useful to read. Also much smaller, compared to the Huggingface Transformers [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) of 1800 lines.
- The final weight loading is in the function `_load_weights` in the above linked Jupyter notebook.
- Next, I want to try running a LoRA/fine-tune on GPT-2 and experiment with using it as a dense feature extractor for retrieval in RAG.
- Then try other, more modern architectures and load those weights again (LLaMa, Qwen, ..)
- MoE, ViT, ..

## Autoencoders
- Implemented simple MNIST autoencoder and ran it through t-SNE: [autoencoder/Autoencoder.ipynb](autoencoder/Autoencoder.ipynb)
- Implemented variational autoencoder from the original paper, trained it on MNIST, generated some images, and ran it through t-SNE: [autoencoder/Variational Autoencoder.ipynb](autoencoder/Variational%20Autoencoder.ipynb)
    - Used latent space dimension of 10. I find the idea of the VAE amongst the most beautiful in machine learning. The fact that we can reduce the space of possibility from intractably high dimensions to 10 dimensions, is beautiful. The idea of the manifold is beautiful. And we can actually *learn* that mapping, so that almost every `torch.randn(10)` maps to a meaningful thing. Now imagine that with molecules..
- Next, I want to try flow models and diffusion models. Excited about the connection to differential equations. Denoising autoencoders first, maybe. 

## Generative Adversarial Networks
- Implemented a basic, dense GAN and trained on MNIST. Generated some nice images. ([GAN/GAN.ipynb](GAN/GAN.ipynb))
- Quite interesting architecture concept and training loop.

## Recurrent Neural Networks
(this is where I started, but the code I'm least proud/sure of)
- Implemented pure RNN (see [rnns/rnn.py](rnns/rnn.py)) which works okay.
- Started implementing LSTM BPTT (backpropagation-through-time). See [rnns/lstm.py](rnns/lstm.py) for more.

The goal is still to support [this tutorial](https://keras.io/examples/generative/lstm_character_level_text_generation/) without using anything out of PyTorch's torch.nn module (e.g. only use torch for fast, GPU capable tensors). Code is written as if PyTorch were NumPy. No autograd is used.

Also want to train an RNN (LSTM or GRU) on ByteDances [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) dataset.

See [RNN_language_model.py](RNN_language_model.py) for an example of how to use the `SimpleRNN` in the `rnn` module.
See [RNN_count.py](RNN_count.py) for an even simpler example, showing how RNN's can very easily learn to count.

See [LSTM_language_model.py](LSTM_language_model.py) and [LSTM_binseq.py](LSTM_binseq.py) for LSTM versions.
The LSTM doesn't function well yet. The language model seems to skew toward learning the endings more (e.g. "out originate out of its opposite?out of its opposite?" as a result, instead of "how could anything"). Will fix soon. (maybe)

### Literature
- [This paper](https://arxiv.org/abs/1503.04069) might shed some light on the troubles I'm having with LSTMs.
