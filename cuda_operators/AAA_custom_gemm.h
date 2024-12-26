
#ifndef __AAA_CUSTOM_GEMM_H_
#define __AAA_CUSTOM_GEMM_H_
#include <cuda_runtime.h>
#include <cublas_v2.h>


// #define OFFSET(row, col, ld) ((row) * (ld) + (col)) //ld是矩阵的列数
// #define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
extern "C" void call_my_sgemm(const float *A, const float *B, float *C, int M, int N, int K);
#endif