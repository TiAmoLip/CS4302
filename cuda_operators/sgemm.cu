#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col)) //ld是矩阵的列数
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// A: 1*1280, B: 1280*5893, C: 1*5893
__global__ void Sgemm(float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
    
    const int BLOCK_SIZE_M = 1;// 一个block中处理的A矩阵行数,或者说一个block处理的A矩阵大小为BLOCK_SIZE_M*BLOCK_SIZE_K
    const int BLOCK_SIZE_N = 128;// 一个block中处理的B矩阵列数,或者说一个block处理的B矩阵大小为BLOCK_SIZE_K*BLOCK_SIZE_N
    const int BLOCK_SIZE_K = 8;// 一个block中处理的A矩阵列数和B矩阵行数,或者说一个block处理的C矩阵大小为BLOCK_SIZE_M*BLOCK_SIZE_N
    // 在上面的设置下，需要开启的线程数就是2048*2048/128/128=16*16=256
    // 在矩阵分块的基础上，我们对每一个block中的A和B矩阵继续进行分块，让每个线程处理rm*rn的子矩阵，那么一个block中的线程数就是BLOCK_SIZE_M*BLOCK_SIZE_N/THREAD_SIZE_Y/THREAD_SIZE_X
    const int TM = 1;
    const int TN = 4;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ float s_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
    float r_c[TN] = {0.0f};
    int load_B_rows = tid >> 5;// 在block中当前线程在B应该load的行
    int load_B_cols = (tid & 31) << 2;// 在block中当前线程在B应该load的列
    // 算了，让所有线程都load A的值吧
    for (int bk = 0; bk < (K+BLOCK_SIZE_K-1)/BLOCK_SIZE_K;++bk) {
        int load_A_cols = bk * BLOCK_SIZE_K;
        // 注意我们只开了32个线程，所以没办法一次就把s_b里的所有东西都load进来
        for (int i=0;i<BLOCK_SIZE_K;++i) {
            if (load_A_cols + i < K) {
                s_a[0][i] = A[OFFSET(0, load_A_cols + i, K)];
            } else {
                s_a[0][i] = 0.0f;
            }
        }
        // 接下来是一个循环将所有的B load进来，但是要注意边界检查
        // 但是这种做贡献的矩阵乘法写法，好像只要在最后数据写回的时候做边界检查就可以了。
        // 这里爆了一个cudaError: Misaligned address错误，原因是我在这里写的时候写到了B的边界外面去了
        // 就是这个循环的问题
        // cnm, 怎么有的float4正常用，有的就报错，服了，直接float吧
        int global_load_b_row_start = load_B_rows + bk * BLOCK_SIZE_K;        
        int global_load_b_col = bx*BLOCK_SIZE_N + load_B_cols;
        // 现在是32个线程，每个线程load1个float,此时load_B_rows = tid/128, load_B_cols = (tid%32)
        // 现在一个row要load4遍
        for (int i=0;i<BLOCK_SIZE_K;++i) {
            int current_load_B_col = (tid % 32) << 2;
            for (int j=0;j<4;++j) {
                if (global_load_b_row_start + i < K && global_load_b_col + j < N) {
                    s_b[i][current_load_B_col+j] = B[OFFSET(global_load_b_row_start + i, global_load_b_col + j, N)];
                } else {
                    s_b[i][current_load_B_col+j] = 0.0f;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k=0;k<BLOCK_SIZE_K;++k) {
            #pragma unroll
            for (int n=0;n<TN;++n) {
                int local_b_col = tx*TN+n;
                if (local_b_col < BLOCK_SIZE_N) {
                    r_c[n] += s_a[0][k] * s_b[k][local_b_col];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j=0;j<TN;j+=4) {
        int global_c_col = bx*BLOCK_SIZE_N + tx*TN+j;
        if (N - global_c_col >= 4) {
            FETCH_FLOAT4(C[OFFSET(0, global_c_col, N)]) = FETCH_FLOAT4(r_c[j]);
        } else {
            for (int i = 0; i < N-global_c_col; ++i) {
                C[OFFSET(0, global_c_col + i, N)] = r_c[j + i];
            }
        }
    }
}

void read_numpy_data(float *A, float *B, float *C, int M, int N, int K) {
    FILE *fp = fopen("A.bin", "rb");
    fread(A, sizeof(float), M * K, fp);
    fclose(fp);
    fp = fopen("B.bin", "rb");
    fread(B, sizeof(float), K * N, fp);
    fclose(fp);
    fp = fopen("C.bin", "rb");
    fread(C, sizeof(float), M * N, fp);
    fclose(fp);
}

void check_result(float *C_cpu, float *C_cuda, int M, int N) {
    for (int i=0;i<M;++i) {
        for (int j=0;j<N;++j) {
            if (abs(C_cpu[i*N+j] - C_cuda[i*N+j]) > 1e-2) {
                printf("Error: C_cpu[%d][%d] = %f, C_cuda[%d][%d] = %f\n", i, j, C_cpu[i*N+j], i, j, C_cuda[i*N+j]);
                return;
            }
        }
    }
    printf("Check result: Correct\n");
}

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

float testCublasPerformance(const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}

int main() {
    const int M = 1, N = 5893, K = 1280;
    float *A_cpu = (float *)malloc(M * K * sizeof(float));
    float *B_cpu = (float *)malloc(K * N * sizeof(float));
    float *C_cpu = (float *)malloc(M * N * sizeof(float));
    float *result_cuda = (float *)malloc(M * N * sizeof(float));
    dim3 DimGrid((N+127)/128, 1, 1);
    dim3 DimBlock(32, 1, 1);
    read_numpy_data(A_cpu, B_cpu, C_cpu, M, N, K);
    float *cuda_A, *cuda_B, *cuda_C;
    cudaMalloc(&cuda_A, M * K * sizeof(float));
    cudaMalloc(&cuda_B, K * N * sizeof(float));
    cudaMalloc(&cuda_C, M * N * sizeof(float));
    cudaMemcpy(cuda_A, A_cpu, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B_cpu, K * N * sizeof(float), cudaMemcpyHostToDevice);
    Sgemm<<<DimGrid, DimBlock>>>(cuda_A, cuda_B, cuda_C, M, N, K);
    cudaMemcpy(result_cuda, cuda_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_result(C_cpu, result_cuda, M, N);
    float sec = testPerformance(Sgemm, DimGrid, DimBlock, M, N, K, 100);
    printf("Kernel time cost: %.6f ms\n", sec);
    sec = testCublasPerformance(M, N, K, 100);
    printf("Cublas time cost: %.6f ms\n", sec);
    cudaError_t err = cudaGetLastError();  
    if (err != cudaSuccess) {  
        printf("CUDA error: %s\n", cudaGetErrorString(err));  
    }
}