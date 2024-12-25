#include <ATen/cuda/CUDABlas.h>

// A: 1*1280, B: 1280*5893, C: 1*5893
// 现在得改写了，A: 5893 * 1280, B: (1280,), C: (5893,)
__global__ void Sgemm_stable(const float * A,const float * B, float * __restrict__ C, int M, int N, int K) {
    
    int row = blockIdx.x * blockDim.x/4 + threadIdx.x/4;  
    float operator_a[4];  
    __shared__ float operator_b[1280];  
    __shared__ float result[128];  // 直接使用一维数组  

    int thread_load_b_cnt = K / 256;  
    for (int i = 0; i < thread_load_b_cnt; i++) {  
        if (threadIdx.x < 256)   
            operator_b[threadIdx.x * thread_load_b_cnt + i] = B[threadIdx.x * thread_load_b_cnt + i];  
    }  
    
    // 初始化result  
    if (threadIdx.x % 4 == 0) {  
        result[threadIdx.x/4] = 0.0f;  
    }  
    __syncthreads();  

    int thread_per_line = blockDim.x / 128;// 4
    int element_per_thread = K / thread_per_line;  
    int thread_calculate_start = (threadIdx.x % 4) * element_per_thread;  

    if (row < M) {  
        float local_sum = 0.0f;  

        for (int i = 0; i < element_per_thread/4; i++) {  
            int idx = 4*i + thread_calculate_start;  
            FETCH_FLOAT4(operator_a) = FETCH_FLOAT4(const_cast<float&>(A[row*K+idx]));

            local_sum += operator_a[0] * operator_b[idx] +   
                         operator_a[1] * operator_b[idx+1] +   
                         operator_a[2] * operator_b[idx+2] +   
                         operator_a[3] * operator_b[idx+3];



        }  
        atomicAdd(&result[threadIdx.x/4], local_sum);  
        
        if (threadIdx.x % 4 == 0) {  
            C[row] = result[threadIdx.x/4];  
        }
    }
}

void call_my_sgemm(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 DimGrid((M+127)/128, 1, 1);
    dim3 DimBlock(512, 1, 1);
    Sgemm_stable<<<DimGrid, DimBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}