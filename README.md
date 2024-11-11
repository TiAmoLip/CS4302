# CS4302
This is final project of CS4302. We are considering to optimize RNN kernels.

We completely avoid using `torchtext` since it will reinstall `pytorch` other than using our own `torch` version.


## Probable Optimizations
1. GRU kernel implementation
2. Cat operator optimization
3. Linear layer.

## Candidate Time analyzer
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
        
        It is worth to note that for a `nn.Linear(a, b)` instance, the shape of weight matrix is `(b, a)`, so maybe I need to implement the transpose operator. I replace `self.fc(out)` to `torch.matmul`, and it seems no influence.


