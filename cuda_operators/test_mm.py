import torch
from torch import nn

# 直接python test_mm.py即可

A = torch.randn(5893, 1280).cuda()
B = torch.randn(1280, 1).cuda()

model = nn.Linear(1280, 1).cuda()

model.weight.data = B.T
model.bias.data = torch.zeros(1).cuda()

C_custom = torch.matmul(A, B) # 建议是编译的时候打开printf
C_ref = model(A) # nn.Linear和matmul使用不同的kernel, 可以据此做检查
print(f"check value result: {torch.allclose(C_custom, C_ref)}, with error :{torch.mean(torch.abs(C_custom - C_ref)).item()}")