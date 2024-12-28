from torch import nn
import torch
import torch.backends
import os
import argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

print("usage: run this script first using original pytorch with parameter --DEBUG_MODE False")
print("Then activate the modified pytorch and run this script again with parameter --DEBUG_MODE True")
parser = argparse.ArgumentParser()
parser.add_argument('--DEBUG_MODE', type=str2bool, default=False)
args = parser.parse_args()
DEBUG_MODE = args.DEBUG_MODE
print(f"DEBUG_MODE: {DEBUG_MODE}")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
embedding_size = 45
seqlen = 11
batch_size = 1
num_layers = 1
hidden_size = 128


if os.path.exists('gru_test.pth'):
    x = torch.load('gru_test.pth').cuda()
    print("load input data")
else:
    x = torch.randn(batch_size, seqlen, embedding_size).cuda()
    torch.save(x, 'gru_test.pth')

gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, batch_first=True).cuda()
if os.path.exists('gru.pth'):
    gru.load_state_dict(torch.load('gru.pth'))
    print("load gru model")
else:
    torch.save(gru.state_dict(), 'gru.pth')

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
# def breakpoint():
#     import os, signal
#     os.kill(os.getpid(), signal.SIGTRAP)
# breakpoint()
output, hidden = gru.forward(x)


if not DEBUG_MODE:
    torch.save(output, 'gru_output.pth')
    torch.save(hidden, 'gru_hidden.pth')
else:
    output_ref = torch.load('gru_output.pth')
    hidden_ref = torch.load('gru_hidden.pth')
    print(f"check output value result: {torch.allclose(output, output_ref)}, with error :{torch.mean(torch.abs(output - output_ref)).item()}")
    print(f"check hidden value result: {torch.allclose(hidden, hidden_ref)}, with error: {torch.mean(torch.abs(hidden - hidden_ref)).item()}" )

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# prof.export_chrome_trace("gru_test.json")
