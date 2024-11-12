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

### Modify pytorch code or custom forward functions?
If the former, I should modify the `aten` operators. (generated by copilet), and I should learn about the function call process first.


## Task 1: Survey Cuda Operators
The operators in `GRU` can be found in the `output/new_kernel_profiler.txt`. We can find several time-consuming kernels like them: `aten::linear`, `aten::addmm` and `aten::gru`.

Hint: in `aten/src/ATen/native/native_functions.yaml` you can find the low-level implementation of top-level operators in the pytorch source code.

### 1.1 Survey the implementation of the cuda kernels in pytorch
Requirements: You should figure out why they can be implemented in parallel, the selection of dimensions for parallelism, the code logic of CUDA kernel functions, and potential optimization space.

### 1.2 CUDA Runtime
Requirements: You should figure out when pytorch called cuda runtime api. Describe its role in the pytorch context.