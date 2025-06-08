import torch
import torch.nn as nn
import math
from torch.nn.functional import softmax

def scaled_dot_product_attention(queries, keys, values, mask=None):
    """
    - Q, K, V each (n_tokens x out_dim)
    - Q @ K.T => (n_tokens x n_tokens) (similarity score)
        - this is actually n_ctx x n_ctx which is 1024x1024 for gpt2
        - that's where the mask comes from for causal attention
        - called "h.0.attn.bias
            - if this were true, then why does every block have it? isn't it the same for all of them?
            - it IS the same..
    - (Q @ K.T) @ V => (QK^T: n_tokens x n_tokens) x (V: n_tokens x out_dim)
                    => (n_tokens x out_dim) returned (attn)
    - dim not changed by sdp
    """
    similarity_score = queries.matmul(keys.T)
    dk = keys.size(-1) # embed_dim
    denom = torch.sqrt(torch.tensor(dk))
    sdp = 1/denom * similarity_score

    if mask is not None:
        n_tokens = keys.size(-2) # <= n_ctx
        mask_val = torch.finfo(sdp.dtype).min # something like -3.4e38 if float32
        masked_attn_weights = torch.where(mask[:n_tokens, :n_tokens], sdp, mask_val)
        sdp = masked_attn_weights

    attn = torch.matmul(softmax(sdp, dim=-1), values)
    return attn

class AttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_ctx): # both equal to embedding_dim
        super().__init__()
        """
        - DIM NOT CHANGED
        - in: n_tokens x embed_dim
        - out: n_tokens x embed_dim
        TODO: batch index -> transpose/matmul correctly
        """
        self.Q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.K_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.V_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, mask=None): # n_tokens x embed_dim
        Q = self.Q_proj(x) # (x: n_tokens x embed_dim) @ (Q_proj: embed_dim x embed_dim)^T -> (n_tokens x embed_dim)
        K = self.K_proj(x) # dim same as x (")
        V = self.V_proj(x) # dim same as x (")
        return scaled_dot_product_attention(Q, K, V, mask=mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, in_dim, out_dim, n_ctx):
        super().__init__()
        self.head_dim = out_dim // n_heads
        self.attn_heads = nn.ModuleList([
            # each head takes in (n_tokens x head_dim)
            # and returns        (n_tokens x head_dim)
            AttentionHead(in_dim, self.head_dim, n_ctx) for _ in range(n_heads)
        ])
        self.register_buffer(
            "mask",
            torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.bool)),
            persistent=False
        )

        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        n_tokens, in_dim = x.shape
        heads = torch.concat([head(x, self.mask) for head in self.attn_heads], dim=-1)
        # = n_tokens x (head_dim*n_heads = out_dim = embed_dim)
        out = self.out_proj(heads)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, in_dim, out_dim, n_ctx):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, in_dim, out_dim, n_ctx) # returns in_dim x dk*n_heads
        self.ln_1 = nn.LayerNorm(out_dim)
        self.ln_2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(in_dim * 4, in_dim)
        )

    def forward(self, x):
        residual_1 = x
        x = self.layernorm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + residual_1

        residual_2 = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual_2
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, num_heads, vocab_size, context_len, embedding_dim):
        super().__init__()
        self.ctx_len = context_len
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(context_len, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.ln_f = nn.LayerNorm(embedding_dim) # apparently need this for GPT-2

        self.blocks = nn.Sequential(*[
            TransformerBlock(num_heads, embedding_dim, embedding_dim, context_len) for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        seq_len = x.shape[-1] # == n_tokens in attention
        assert seq_len <= self.ctx_len
        positions = torch.arange(seq_len) # 0, 1, 2, .., seq_len-1

        # https://jalammar.github.io/illustrated-gpt2/#:~:text=Part%20of%20the%20trained%20model%20is%20a%20matrix%20that%20contains%20a%20positional%20encoding%20vector%20for%20each%20of%20the%201024%20positions%20in%20the%20input.
        embeddings = self.token_embedding(x) + self.positional_embedding(positions)

        x = self.dropout(embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.linear(x) # this one isn't in https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1069

        return self.softmax(x)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[-context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[-1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=0)  # (batch, n_tokens+1)

    return idx
