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

但是就算这样，也不能正常用gdb，我看了一下pytorch forum，也有人问相关问题，回答是你要用python的调试版才能进入C++代码。于是我又去下了python源码开始编译，过程中其实遇到这样的错误
```sh
Python build finished successfully!
The necessary bits to build these optional modules were not found:
_bz2                  _curses               _curses_panel      
_dbm                  _gdbm                 _hashlib           
_lzma                 _sqlite3              _ssl               
_tkinter              _uuid                 readline           
To find the necessary bits, look in setup.py in detect_modules() for the module's name.
The following modules found by detect_modules() in setup.py, have been
built by the Makefile instead, as configured by the Setup files:
_abc                  atexit                pwd                
time                                                           
Could not build the ssl module!
Python requires an OpenSSL 1.0.2 or 1.1 compatible libssl with X509_VERIFY_PARAM_set1_host().
LibreSSL 2.6.4 and earlier do not provide the necessary APIs, 
https://github.com/libressl-portable/portable/issues/381
```
博客上的做法就是直接嗯装包，但是我装了一波包甚至失败了一个curse(这个博客上也有解决办法，但是我后面莫名其妙没这个报错了)，所以我就直接上去make，安装倒是成功了。但是你还要在pytorch里面重新装一遍。


但后来我用非debug版本的python也能gdb，debug版本的优势在于，你用`gdb --args python-debug test.py`进去之后，直接list可以看到python的c调用。首先我`b xx.py`并不能成功设置断点，而`b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/torch/csrc/autograd/python_variable.cpp:1461`是可以的(当然py文件就是一个加法), pytorch forum里面人教了一个很神奇的方法:
```py
def breakpoint():
    import os, signal
    os.kill(os.getpid(), signal.SIGTRAP)
# your code...
breakpoint()  # set a breakpoint
# your code...
```

突然发现pdb其实可以和gdb一块存在

可能的一些breakpoint(老是step进去就晕了)
```py
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/torch/csrc/autograd/python_variable.cpp:1461
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/torch/csrc/autograd/python_variable.cpp:1601
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/native/LinearAlgebra.cpp:460
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/torch/csrc/autograd/generated/python_torch_functions_0.cpp:4070 # 这个是matmul的时候用到的
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/native/LinearAlgebra.cpp:1834 # matmul哈哈哈哈哈哈哈哈哈终于找到你了小子
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/native/LinearAlgebra.cpp:1710 # 相比上一个更细节
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/torch/csrc/autograd/generated/VariableType_3.cpp:8174 # 更细节,但这个断点第一次run的时候打不上。
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/native/cuda/Blas.cpp:420 # 终于跳到cuda了
/root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/cuda/CUDABlas.cpp:353 # CUDABlas具体实现找不到
# 8174之后还要第三次dispatch
```
善用finish和step 10, 最后真能找到kernel. step 1基本不可能，step 20和10基本不会错过。

## Task 1: Survey Cuda Operators
The operators in `GRU` can be found in the `output/new_kernel_profiler.txt`. We can find several time-consuming kernels like them: `aten::linear`, `aten::addmm` and `aten::gru`.

Hint: in `aten/src/ATen/native/native_functions.yaml` you can find the low-level implementation of top-level operators in the pytorch source code.

### 1.1 Survey the implementation of the cuda kernels in pytorch
Requirements: You should figure out why they can be implemented in parallel, the selection of dimensions for parallelism, the code logic of CUDA kernel functions, and potential optimization space.

To find the corresponding cuda kernel, we need to make great use of `search` in vscode sidebar. From project pdf, we extract `castPyCFunctionWithKeywords(THPVariable_mm)` and find `python_torch_functionsEverything.cpp`. In this file, we can find the corresponding binding functions of all the mentioned operators above.

以`mm`作为一个研究特例，进入`THPVariable_mm`函数，可以发现里面实际上根据你传的参数的不同，他会走好几条算子的路线，但基本是大差不差的。但是一路走下去点到`op.call`就找不到更具体的实现了，继续看文档

先研究seq2seq里的encoder模块干的事情，这里的算子主要包括embedding、dropout和rnn。通过上面提到的breakpoint和print字符来确定入口，用finish命令加速确定。

但我跑到这里
```
call_function (kwnames=0x0, oparg=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>, 
    pp_stack=<synthetic pointer>, tstate=0x1066f30) at /usr/local/src/conda/python-3.9.20/Python/ceval.c:5083
_PyEval_EvalFrameDefault (tstate=<optimized out>, f=0x7f48ea87e040, throwflag=<optimized out>) at /usr/local/src/conda/python-3.9.20/Python/ceval.c:5088
5088    in /usr/local/src/conda/python-3.9.20/Python/ceval.c
```
他就直接完成forward函数了，所以finish还是谨慎用。

Embedding:
```bash
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/torch/csrc/autograd/generated/python_torch_functions_0.cpp:2744 # 函数名torch::autograd::THPVariable_embedding，所以后面说不定不用每次都step进去找，但dropout和rnn没找到对应的thP

b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/core/boxing/KernelFunction_impl.h:54 #这是众多算子都要过的一行



关键步骤是index_select, 猜测在这里ctrl-f:
/root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/native/TensorShape.cpp

# 为了加速找到算子，建议 b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/core/boxing/KernelFunction_impl.h:66
embedding 的cu:
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/native/cuda/Indexing.cu:1167
剩下的说实话你可以ctrl+leftclick跳转了. 基本就在一个cu里. 不过gdb貌似无法调试cuda
```

GRU
```bash
b /root/ZhangRui/CS4302/Final/pytorch-v1.12.1/aten/src/ATen/native/cudnn/RNN.cpp:1641
```

#### embedding Indexing.cu
```cpp
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<T, IndexType> src,
                                      cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
/*
indices.data是指针，dstIndex是int，先将dstIndex

目前已知indices[0]是13,src是embedding表，dst是希望的值。我希望dst从0开始，src则要根据传入的indices来选择
而python源文件里，传入的下标是2,56,13,11,6,175,106,9,15,75,0,4,3
cu文件里的src和dst都是(1,256), 光看kernel不行，还要注意这个函数index_select_out_cuda_impl
*/


  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}
```


友情链接:
- pytorch dispatcher: http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/
- pytoch 源码解析(csdn): https://blog.csdn.net/tanmx219/article/details/86705952
- pytorch C层面调试: https://blog.csdn.net/tanmx219/article/details/86762506 注意，需要你先clean掉之前编译的release版本之后再setup build develop
- https://blog.csdn.net/Yong_Qi2015/article/details/123415003 手写算子提速的例子
- 代码结构与编译: https://www.luokai.tech/posts/build/

所以可以直接搜一下有无现成的教程

### 1.2 CUDA Runtime
Requirements: You should figure out when pytorch called cuda runtime api. Describe its role in the pytorch context.