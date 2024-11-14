
import torch
import os, signal

# def breakpoint():
#     os.kill(os.getpid(), signal.SIGTRAP)
# batch_size = 32
# seq_len = 100
# embedding_size = 256
# a = torch.rand((seq_len, batch_size, embedding_size))

# layer = torch.nn.LSTM(input_size=embedding_size, hidden_size=512, num_layers=2, batch_first=False)

# res = layer.forward(a)

def test_add():
    a = torch.Tensor(3,2).uniform_(-1,1).cuda()
    b = torch.Tensor(3,2).uniform_(-1,1).cuda().T
    # breakpoint()
    print("enter matmul python code")
    torch.matmul(a,b)
test_add()
# print(res[0].shape)