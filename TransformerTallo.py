import torch
from torch import nn, device, manual_seed, rand
from tqdm import tqdm

from TransformerEncoderTallo import TransformerEncoder as Encoder
from TransformerDecoderTallo import TransformerDecoder as Decoder

class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_model: int,
            num_attention_heads: int,
            units_hidden_layer: int,
            dropout: float,
            activation: nn.Module = nn.ReLU(),
            mask: bool = True,
            device: device = device("cpu")):

        super().__init__()
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=dim_model,
            num_heads=num_attention_heads,
            units_hidden_layer=units_hidden_layer,
            dropout=dropout,
            activation=activation,
            mask=mask,
            device=device)

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=dim_model,
            num_heads=num_attention_heads,
            units_hidden_layer=units_hidden_layer,
            dropout=dropout,
            activation=activation,
            mask=mask,
            device=device)

    def forward(self, src, target):

        output = torch.empty(size=src.size())

        for i, element in enumerate(tqdm(src)):
            input = torch.reshape(element, (1, element.size(0), element.size(1)))

            step_1 = self.encoder(input)

            for j in range(src.size(1)):
                trgt = torch.reshape(target[i][j], (1, target.size(1), target.size(2)))


                out = self.decoder(src=trgt, memory=step_1)

            output[i] = out

        return output
