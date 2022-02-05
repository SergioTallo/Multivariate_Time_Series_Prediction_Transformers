import math
import torch

from torch import tensor, nn, uint8, cat, device, float32, sin, cos, div

import torch.nn.functional as f
import multiprocessing as mp


def generate_attention_mask(size: tuple) -> tensor:
    # https://pytorch.org/docs/stable/generated/torch.ones.html
    # https://pytorch.org/docs/stable/generated/torch.triu.html

    # Step 1: Create a mask matrix with the same shape as the query and key tensors and values -inf
    # [batch_size, sequence_len, sequence_len]
    mask = torch.ones(size, dtype=uint8) * float('-inf')

    # Step 2: Set the diagonal of the mask matrix to 0
    # [batch_size, sequence_len, sequence_len]
    mask = mask.triu(diagonal=1)

    return mask


def scaled_dot_product_attention(query: tensor, key: tensor, value: tensor, mask: tensor = None) -> tensor:
    # https://pytorch.org/docs/stable/generated/torch.bmm.html
    # https://pytorch.org/docs/stable/generated/torch.transpose.html
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax

    # Step 1: matrix multiplication between query and key transpose
    # Transpose dimension 2 and 3
    key = key.transpose(1, 2)
    step_1 = torch.bmm(query, key)
    # Dimensions after step_ 1 [batch_size, sequence_len, sequence_len]

    # Step 2: Scale the matrix by dividing by the square root of the key dimension (number of features)
    d_k = query.size(-1)
    step_2 = step_1 / math.sqrt(d_k)
    # Dimensions after step_ 2 [batch_size, sequence_len, sequence_len]

    # Step 3: Apply the mask to the attention matrix if mask is not None
    if mask is not None:
        step_3 = step_2 + mask
    else:
        step_3 = step_2

    # Step 4: Apply softmax to the attention matrix
    step_4 = f.softmax(step_3, dim=-1)
    # Dimensions after step_4 [batch_size, sequence_len, sequence_len]

    # Step 5: Multiply the attention matrix with the values
    step_5 = torch.bmm(step_4, value)
    # Dimensions after step_5 [batch_size, sequence_len, number of features]

    return step_5


def feed_forward_layer(d_model: int, units_hidden_layer: int, activation: nn.Module) -> nn.Module:
    # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html

    # d_model: dimension of the model (number of features)
    # units_hidden_layer: number of units in the hidden layer
    # activation: activation function

    layer = nn.Sequential(
        nn.Linear(d_model, units_hidden_layer),
        activation,
        nn.Linear(units_hidden_layer, d_model),
    )

    return layer


# todo: comment positional encoding
def positional_encoding(seq_len: int, dim_model: int, device: device = device("cpu")) -> tensor:
    pos = torch.arange(seq_len, dtype=float32, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=float32, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (torch.div(dim, dim_model, rounding_mode='floor')))

    position_encoding = torch.where(dim.long() % 2 == 0, sin(phase), cos(phase))

    return position_encoding


class AttentionHead(nn.Module):

    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    def __init__(self, d_model: int, d_key: int, d_query: int):
        # d_model: dimension of the model (number of features)
        # d_k: dimension of the key (max(dim_model // num_heads, 1))
        # d_q: dimension of the query (max(dim_model // num_heads, 1))

        super().__init__()

        self.query = nn.Linear(d_model, d_query)
        self.key = nn.Linear(d_model, d_key)
        self.value = nn.Linear(d_model, d_key)

    def forward(self, query: tensor, key: tensor, value: tensor, mask: bool = False) -> tensor:

        # Step 1: Apply the linear layers to the query, key and value
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Step 1.1: Generate the attention mask
        # dimensions of the mask [batch_size, sequence_len, sequence_len]
        if mask:
            mask = generate_attention_mask((query.size(0), query.size(1), key.size(1)))

        # Step 2: Apply scaled dot product attention
        step_2 = scaled_dot_product_attention(query, key, value, mask)

        return step_2


class MultiHeadAttentionLayer(nn.Module):

    # https://pytorch.org/docs/stable/nn.html#torch.nn.ModuleList
    # https://pytorch.org/docs/stable/generated/torch.cat.html

    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 d_key: int,
                 d_query: int,
                 mask: bool = False):
        # d_model: dimension of the model (number of features)
        # d_k: dimension of the key (max(number of features // num_heads, 1))
        # d_q: dimension of the query (max(number of features // num_heads, 1))

        super().__init__()

        self.mask = mask

        # Step 1: Create the attention heads (so many as num_heads)
        self.attention_heads = nn.ModuleList([AttentionHead(d_model, d_key, d_query) for _ in range(num_heads)])

        # Step 3: Create a linear layer to combine the heads
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, query: tensor, key: tensor, value: tensor, mask: bool = False) -> tensor:
        # Step 1: Apply the attention heads to the query, key and value and concatenate the results
        step_1 = cat([h(query, key, value, self.mask) for h in self.attention_heads], dim=-1)

        # Step 2: Apply the final linear layer to the concatenated results
        step_2 = self.linear_layer(step_1)

        return step_2


class Residual(nn.Module):

    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: tensor) -> tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.

        # Step 1: Apply the sublayer
        step_1 = self.dropout(self.sublayer(*tensors))

        # Step 2: Add input and output of sublayer
        step_2 = step_1 + tensors[0]

        # Step 3: Apply the layer norm
        step_3 = self.norm(step_2)

        return step_3


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 units_hidden_layer: int,
                 dropout: float = 0.1,
                 activation: nn.Module = nn.ReLU(),
                 mask: bool = False):
        # d_model: dimension of the model (number of features)
        # num_heads: number of attention heads in the multi-head attention layer
        # units_hidden_layer: number of units in the hidden layer of the feed forward layer
        # dropout: dropout probability

        # d_k: dimension of the key (max(number of features // num_heads, 1))
        # d_q: dimension of the query (max(number of features // num_heads, 1))

        super().__init__()

        dim_q = dim_k = int(div(d_model, num_heads, rounding_mode='trunc'))

        self.attention = Residual(
            MultiHeadAttentionLayer(num_heads=num_heads, d_model=d_model, d_query=dim_q, d_key=dim_k, mask=mask),
            dimension=d_model, dropout=dropout)

        self.feed_forward = Residual(
            feed_forward_layer(d_model=d_model, units_hidden_layer=units_hidden_layer, activation=activation),
            dimension=d_model, dropout=dropout)

    def forward(self, src: tensor) -> tensor:
        # Step 1: Apply the attention layer
        step_1 = self.attention(src, src, src)

        # Step 2: Apply the feed forward layer
        step_2 = self.feed_forward(step_1)

        return step_2


class TransformerEncoder(nn.Module):

    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 units_hidden_layer: int,
                 dropout: float = 0.1,
                 mask: bool = False,
                 activation: nn.Module = nn.ReLU(),
                 device: device = device("cuda" if torch.cuda.is_available() else "cpu")):

        # num_layers: number of encoder layers
        # d_model: dimension of the model (number of features)
        # num_heads: number of attention heads in the multi-head attention layer
        # units_hidden_layer: number of units in the hidden layer of the feed forward layer
        # dropout: dropout probability
        # mask: whether to mask the attention layer
        # activation: activation function to use in the feed forward layer
        # device: device to use

        super().__init__()

        self.d_model = d_model
        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, units_hidden_layer=units_hidden_layer,
                                     dropout=dropout, mask=mask, activation=activation) for _ in range(num_layers)])

    def forward(self, src: tensor) -> tensor:
        # The dimension of the model has to be equal as the number of features of the input tensor
        assert src.shape[-1] == self.d_model, \
            "The last dimension of the input tensor (number of features) must be equal to the dimension of the model"

        # Step 1: Apply the positional encoding
        src += positional_encoding(seq_len=src.size(1), dim_model=self.d_model, device=self.device)

        # Step 2: Apply the encoder layers
        for layer in self.layers:
            src = layer(src)

        return src
