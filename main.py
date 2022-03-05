from time import sleep

import torch
from torch import manual_seed, rand, device, nn
import multiprocessing as mp
import transformers


from TransformerTallo import Transformer

from tqdm import tqdm

print("Number of processors: ", mp.cpu_count())

print('First')
manual_seed(0)
src = rand(10, 30, 18)

print(src.size())

model = Transformer(num_encoder_layers=1,
                    num_decoder_layers=1,
                    dim_model=18,
                    num_attention_heads=1,
                    units_hidden_layer=2048,
                    dropout=0.1,
                    activation=nn.ReLU(),
                    mask=True,
                    device=device("cpu"))(src, src)

print(model.size())

for i in range(src.size(1)):
    print(f'Batch: {i}')
    print(src[i][0])
    print(model[i][0])
