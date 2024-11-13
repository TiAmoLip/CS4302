# CS4302
This is final project of CS4302. We are considering to optimize RNN kernels.

We completely avoid using `torchtext` since it will reinstall `pytorch` other than using our own `torch` version.

## Analysis
### Possible Optimizations
1. GRU kernel implementation
2. Cat operator optimization
3. Linear layer.

### Candidate Time analyzer
1. torchprofiler
2. torch.cuda.events
3. Nvidia nsight systems (but difficult to figure out each kernel)


From torchProfiler I find: The largest bottleneck:
std::enable_if<!T7, void>::type internal::gemvx::kernel, which seems to be unable for me to optimize. 

Other bottleneck:
1. aten::linear
2. aten::addmm
3. aten::gru
4. aten::_cudnn_rnn

Operators to implement:
1. encoder:
    - GRU forward:
        - input shape: `(seq_len, bs, embed_dim)`
        - output `output`: `(seq_len, bs, hidden_dim)`
        - output `hidden`: `(nlayers, bs, hidden_dim)`
2. decoder:
    - GRU forward: 
        - input `hidden`: `(1, bs, hidden_dim)`
        - input `embed`: `(1, bs, hidden_dim+embed_dim)` 
        - output `output`: `(1, bs, hidden_dim)`
        - output `hidden`: `(1, bs, hidden_dim)`
    - Linear forward:
        - input shape: `(bs, hidden_dim*2+embed_dim)`
        - output shape: `(bs, vocab_size)`
        
        It is worth to note that for a `nn.Linear(a, b)` instance, the shape of weight matrix is `(b, a)`, so maybe I need to implement the transpose operator. I replace `self.fc(out)` to `torch.matmul`, and it seems no influence for time consumption.

### Operator Fusion:
Some complex code in python can be fused into one operator in C++.

```python
slow case: out = eps + F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w.unsqueeze(1), groups=C) # from https://blog.csdn.net/qq_29788741/article/details/130463197
```

### Modify pytorch code or custom forward functions?
If the former, I should modify the `aten` operators. (generated by copilet), and I should learn about the function call process first.


## 记录一下配环境的过程
第一次从pytorch源码配环境，去release里面找到1.12.1, 下载197MB的那个包。3090 + nvcc11.6(另一个4090 就不能用1.12.1), 先去`pytorch-v1.12.1/tools/setup_tools/env.py`里面添加如下代码:
```py
os.environ['DEBUG'] = '1'
os.environ['CMAKE_BUILD_TYPE'] = 'Debug'
```
然后执行`Debug=1 python setup.py install`, 就可以装出来debug版本的pytorch了。看网上说的，你不装debug版本，调试的时候进不去C++代码，只能看到python代码。

注意不要在`pytorch-v1.12.1`目录下面进入python然后import torch，会报错。
```py
cd ..
python
import torch
print(torch.version.debug) # True
```

## Task 1: Survey Cuda Operators
The operators in `GRU` can be found in the `output/new_kernel_profiler.txt`. We can find several time-consuming kernels like them: `aten::linear`, `aten::addmm` and `aten::gru`.

Hint: in `aten/src/ATen/native/native_functions.yaml` you can find the low-level implementation of top-level operators in the pytorch source code.

### 1.1 Survey the implementation of the cuda kernels in pytorch
Requirements: You should figure out why they can be implemented in parallel, the selection of dimensions for parallelism, the code logic of CUDA kernel functions, and potential optimization space.

To find the corresponding cuda kernel, we need to make great use of `search` in vscode sidebar. From project pdf, we extract `castPyCFunctionWithKeywords(THPVariable_mm)` and find `python_torch_functionsEverything.cpp`. In this file, we can find the corresponding binding functions of all the mentioned operators above.

以`mm`作为一个研究特例，进入`THPVariable_mm`函数，可以发现里面实际上根据你传的参数的不同，他会走好几条算子的路线，但基本是大差不差的。但是一路走下去点到`op.call`就找不到更具体的实现了，继续看文档

友情链接:
- pytorch dispatcher: http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/
- pytoch 源码解析(csdn): https://blog.csdn.net/tanmx219/article/details/86705952
- pytorch C层面调试: https://blog.csdn.net/tanmx219/article/details/86762506 注意，需要你先clean掉之前编译的release版本之后再setup build develop
- 代码结构与编译: https://www.luokai.tech/posts/build/
所以可以直接搜一下有无现成的教程

### 1.2 CUDA Runtime
Requirements: You should figure out when pytorch called cuda runtime api. Describe its role in the pytorch context.