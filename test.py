
import torch

batch_size = 32
seq_len = 100
embedding_size = 256
a = torch.rand((seq_len, batch_size, embedding_size))

layer = torch.nn.LSTM(input_size=embedding_size, hidden_size=512, num_layers=2, batch_first=False)

res = layer.forward(a)

print(res[0].shape)