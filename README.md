## 文件说明

### output
存放pytorch profiler跑出的各个kernel的运行时间。其中, `custom_kernel_result.txt`是我们的两个kernel替换原版之后的运行时间。`original_kernel_profiler.txt`是原版kernel的运行时间。

### vocabs
这是用于平替torchtext的词表，主要因为对应版本的torchtext一旦pip安装就会尝试自动卸载已经装好的pytorch1.12.1.

### cuda_operators

- `RNN.cu`存放我们对gru kernel的改装，他应该放到pytorch源码的`aten/src/ATen/native/cuda/RNN.cu`中.
- `AAA_custom_gemm.cu`, `AAA_custom_gemm.h`, `CUDABlas.cpp`, `CUDA_Blas.h`是我们对矩阵乘法的改造，对应位置分别是`aten/src/ATen/native/cuda/AAA_custom_gemm.cu`, `aten/src/ATen/cuda/AAA_custom_gemm.h`. `aten/src/ATen/cuda/CUDABlas.cpp`, `aten/src/ATen/cuda/CUDABlas.h`.

- `test_gru.py`和`test_mm.py`则是正确性检查. 对应的使用方法在文件中

- `sgemm.cu`则是我们提到的几版matmul的改写，运行方法首先要生成测试数据, 即利用`test_data_generation.py`, 然后在`sgemm.cu`里将你想运行的版本改名字, 最后运行`nvcc -o sgemm sgemm.cu -lcublas && ./sgemm && rm -f sgemm`.

### 最外层
- main.py 是我们的主程序，运行方法是`python main.py`，其中的参数可以在文件中修改。注意他会自动运行profiler，可选参数是保存的profiler的名字. 此外当完全运行完之后他会执行一次检查，即检查解码后的文本和我提前用原本pytorch跑出的结果是否相符。理论上我们的kernel正确率是0.9990... 对应的reference在`vocabs/original_predictions.txt`中

- `tut2-model.pt`是我们用来推理的模型参数，是原本的笔记本跑出来的最后一个iteration的Seq2Seq模型参数。

- `backtrack.md`是我用gdb追溯的调用栈方便我理解

- `analyze.md`是我整个大作业的思考历程。可能有点乱