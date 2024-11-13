#include "custom_matmul.h"
#include <torch/extension.h>

void torch_launch_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  launch_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), A.size(0), B.size(1), A.size(1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_matmul", &torch_launch_matmul, "Launch matmul");
}