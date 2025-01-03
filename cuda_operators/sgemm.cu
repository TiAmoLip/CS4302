#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col)) //ld是矩阵的列数
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// A: 1*1280, B: 1280*5893, C: 1*5893
__global__ void Sgemm(float * A,float * B, float * C, int M, int N, int K) {
    
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
    if (tid==0) printf("A[0][0] = %f, B[0][0] = %f\n", A[0], B[0]);
    __shared__ float s_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
    float operator_b[TN];// 在最后计算的时候，将东西load到register，然后和s_a做乘法，实测从162变成158
    float r_c[TN] = {0.0f};
    int load_B_rows = tid >> 5;// 在block中当前线程在B应该load的行
    int load_B_cols = (tid & 31) << 2;// 在block中当前线程在B应该load的列
    // 算了，让所有线程都load A的值吧
    #pragma unroll
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
        // __syncthreads();
        // 接下来是一个循环将所有的B load进来，但是要注意边界检查
        // 但是这种做贡献的矩阵乘法写法，好像只要在最后数据写回的时候做边界检查就可以了。
        // 这里爆了一个cudaError: Misaligned address错误，原因是我在这里写的时候写到了B的边界外面去了
        // 就是这个循环的问题
        // cnm, 怎么有的float4正常用，有的就报错，服了，直接float吧
        int global_load_b_row_start = load_B_rows + bk * BLOCK_SIZE_K;        
        int global_load_b_col = bx*BLOCK_SIZE_N + load_B_cols;
        // 现在是32个线程，每个线程load1个float,此时load_B_rows = tid/128, load_B_cols = (tid%32)
        // 现在一个row要load4遍
        int current_load_B_col = tid << 2;
        #pragma unroll
        for (int i=0;i<BLOCK_SIZE_K;++i) {
            
            #pragma unroll
            for (int j=0;j<4;++j) {
                if (global_load_b_row_start + i < K && global_load_b_col + j < N) {
                    s_b[i][current_load_B_col+j] = B[OFFSET(global_load_b_row_start + i, global_load_b_col + j, N)];
                } else {
                    s_b[i][current_load_B_col+j] = 0.0f;
                }
            }
        }
        __syncthreads();
        // 我试一下load到寄存器里再做乘法会不会更快

        // 但是这里用float4更快
        #pragma unroll
        for (int k=0;k<BLOCK_SIZE_K;++k) {
            FETCH_FLOAT4(operator_b) = FETCH_FLOAT4(s_b[k][tx*TN]);
            #pragma unroll
            for (int n=0;n<TN;++n) {
                r_c[n] += s_a[0][k] * operator_b[n];
            }
        }
        // __syncthreads();
    }
    // 这个好奇怪啊，用float4反而更慢了
    #pragma unroll
    for (int j=0;j<TN;++j) {
        int global_c_col = bx*BLOCK_SIZE_N + tx*TN+j;
        if (global_c_col < N) {
            C[OFFSET(0, global_c_col, N)] = r_c[j];
        }
    }
}

// A: 5893 * 1280, B: (1280, 1), C: (5893,1)

__global__ void Sgemm_naive_stable0(float * A,float * B, float * C, int M, int N, int K) {
    // 最native的实现，每个线程负责一个row，128线程
    // 此外下面所有的都用了unroll展开和SIMD
    // 1280:426ms, 用了unroll是425ms
    // 768: 3往后所有都是25ms, 0~2没测
    // 推荐128线程
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;

        // for (int i = 0; i < K; i++) {
        //     sum+=A[row*K+i]*B[i];
        // }
        for (int i = 0; i < K/4; i++) {
            sum += A[row*K+4*i] * B[4*i] + A[row*K+4*i+1] * B[4*i+1] + A[row*K+4*i+2] * B[4*i+2] + A[row*K+4*i+3] * B[4*i+3];
        }
        C[row] = sum;
    }
}

__global__ void Sgemm_naive_stable1(float * A,float * B, float * C, int M, int N, int K) {
    // 每个线程负责一个row，128线程
    // 此外下面所有的都用了unroll展开
    // 1280:112ms

    // 128线程
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float operator_a[4];
    float operator_b[4];
    if (row < M) {
        float sum = 0.0f;

        for (int i = 0; i < K/4; i++) {
            FETCH_FLOAT4(operator_a) = FETCH_FLOAT4(const_cast<float&>(A[row*K+4*i]));
            FETCH_FLOAT4(operator_b) = FETCH_FLOAT4(const_cast<float&>(B[4*i]));
            sum += operator_a[0] * operator_b[0] + operator_a[1] * operator_b[1] + operator_a[2] * operator_b[2] + operator_a[3] * operator_b[3];
        }
        C[row] = sum;
    }
}

__global__ void Sgemm_naive_stable2(float * A,float * B, float * C, int M, int N, int K) {
    // 和上面setting几乎一样，区别是引入了预取
    // 1280:227ms
    // 128线程
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float operator_a[2][4];
    float operator_b[2][4];
    if (row < M) {
        float sum = 0.0f;
        int write_idx = 0;
        FETCH_FLOAT4(operator_a[1-write_idx]) = FETCH_FLOAT4(A[row*K]);
        FETCH_FLOAT4(operator_b[1-write_idx]) = FETCH_FLOAT4(B[0]);
        for (int i = 0; i < K/4-1; i++) {
            int load_idx = 1-write_idx;
            FETCH_FLOAT4(operator_a[write_idx]) = FETCH_FLOAT4(A[row*K+(i+1)*4]);
            FETCH_FLOAT4(operator_b[write_idx]) = FETCH_FLOAT4(B[(i+1)*4]);
            sum += operator_a[load_idx][0] * operator_b[load_idx][0] + operator_a[load_idx][1] * operator_b[load_idx][1] + operator_a[load_idx][2] * operator_b[load_idx][2] + operator_a[load_idx][3] * operator_b[load_idx][3];
            
            write_idx = 1 - write_idx;
        }
        int load_idx = 1-write_idx;
        sum += operator_a[load_idx][0] * operator_b[load_idx][0] + operator_a[load_idx][1] * operator_b[load_idx][1] + operator_a[load_idx][2] * operator_b[load_idx][2] + operator_a[load_idx][3] * operator_b[load_idx][3];

        C[row] = sum;
    }
}
__global__ void Sgemm_naive_stable3(float * A, float * B, float * C, int M, int N, int K) {
    // 1280:53ms, 512线程，4个线程负责1行
    // 加入shared memory
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
            FETCH_FLOAT4(operator_a) = FETCH_FLOAT4(A[row*K+idx]);

            local_sum += operator_a[0] * operator_b[idx] +   
                         operator_a[1] * operator_b[idx+1] +   
                         operator_a[2] * operator_b[idx+2] +   
                         operator_a[3] * operator_b[idx+3];



        }  
        // 使用原子加法将局部和累加到result  
        atomicAdd(&result[threadIdx.x/4], local_sum);  
        // __syncthreads();  

        // 只有第一个线程写最终结果  
        if (threadIdx.x % 4 == 0) {  
            C[row] = result[threadIdx.x/4];  
        }  
        // __syncthreads();  
    }  
}

__global__ void Sgemm_naive_stable4(float * A, float * B, float * C, int M, int N, int K) {
    // 和3几乎一样，区别是引入了预取
    // 1280:73ms
    
    int row = blockIdx.x * blockDim.x/4 + threadIdx.x/4;  
    
    // 预取数据的变量  
    float operator_a[2][4];    
    
    __shared__ float operator_b[1280];  
    __shared__ float result[128];  

    int thread_load_b_cnt = K / 256;  
    for (int i = 0; i < thread_load_b_cnt; i++) {  
        if (threadIdx.x < 256)   
            operator_b[threadIdx.x * thread_load_b_cnt + i] = B[threadIdx.x * thread_load_b_cnt + i];  
    }  
    
    if (threadIdx.x % 4 == 0) {  
        result[threadIdx.x/4] = 0.0f;  
    }  
    __syncthreads();  

    int thread_per_line = blockDim.x / 128;  
    int element_per_thread = K / thread_per_line;  
    int thread_calculate_start = (threadIdx.x % 4) * element_per_thread;  

    if (row < M) {  
        float local_sum = 0.0f;  

        // 预取第一组数据
        int write_idx = 0;
        FETCH_FLOAT4(operator_a[write_idx]) = FETCH_FLOAT4(A[row*K + thread_calculate_start]);  
        
        for (int i = 0; i < element_per_thread/4 - 1; i++) {  
            int idx = 4*i + thread_calculate_start;  
            write_idx = 1 - write_idx;
            // 预取下一组数据  
            FETCH_FLOAT4(operator_a[write_idx]) = FETCH_FLOAT4(A[row*K + idx + 4]);  
            
            // 使用当前数据计算  
            local_sum += operator_a[1-write_idx][0] * operator_b[idx] +   
                         operator_a[1-write_idx][1] * operator_b[idx+1] +   
                         operator_a[1-write_idx][2] * operator_b[idx+2] +   
                         operator_a[1-write_idx][3] * operator_b[idx+3];  
            
            // 交换数据  
            // memcpy(operator_a, operator_a_next, sizeof(float) * 4);  
        }  

        // 处理最后一组数据  
        int last_idx = 4 * (element_per_thread/4 - 1) + thread_calculate_start;  
        local_sum += operator_a[write_idx][0] * operator_b[last_idx] +   
                     operator_a[write_idx][1] * operator_b[last_idx+1] +   
                     operator_a[write_idx][2] * operator_b[last_idx+2] +   
                     operator_a[write_idx][3] * operator_b[last_idx+3];
                     
        
        // 原子加法累加结果  
        atomicAdd(&result[threadIdx.x/4], local_sum);  
        __syncthreads();  

        // 只有第一个线程写最终结果  
        if (threadIdx.x % 4 == 0) {  
            C[row] = result[threadIdx.x/4];  
        }  
        __syncthreads();  
    }  
}
__global__ void Sgemm_naive_stable5(float * A, float * B, float * C, int M, int N, int K) {  
    // 和3几乎一样，区别是加大了unroll的size
    // 1280: 55ms
    int row = blockIdx.x * blockDim.x/4 + threadIdx.x/4;  
    float operator_a[8];  
    __shared__ float operator_b[1280];  
    __shared__ float result[128];  // 直接使用一维数组  

    int thread_load_b_cnt = K / (blockDim.x/2);  
    for (int i = 0; i < thread_load_b_cnt; i++) {  
        if (threadIdx.x < blockDim.x/2)   
            operator_b[threadIdx.x * thread_load_b_cnt + i] = B[threadIdx.x * thread_load_b_cnt + i];  
    }  
    
    // 初始化result  
    if (threadIdx.x % 4 == 0) {  
        result[threadIdx.x/4] = 0.0f;  
    }  
    __syncthreads();  

    int thread_per_line = blockDim.x / 128;  
    int element_per_thread = K / thread_per_line;  
    int thread_calculate_start = (threadIdx.x % 4) * element_per_thread;  

    if (row < M) {  
        float local_sum = 0.0f;  

        for (int i = 0; i < element_per_thread/8; i++) {  
            int idx = 8*i + thread_calculate_start;  
            FETCH_FLOAT4(operator_a) = FETCH_FLOAT4(A[row*K+idx]);
            FETCH_FLOAT4(operator_a[4]) = FETCH_FLOAT4(A[row*K+idx+4]);
            
            local_sum += operator_a[0] * operator_b[idx] +   
                         operator_a[1] * operator_b[idx+1] +   
                         operator_a[2] * operator_b[idx+2] +   
                         operator_a[3] * operator_b[idx+3] + 
                            operator_a[4] * operator_b[idx+4] +
                            operator_a[5] * operator_b[idx+5] +
                            operator_a[6] * operator_b[idx+6] +
                            operator_a[7] * operator_b[idx+7];
        }  
        
        // 使用原子加法将局部和累加到result  
        atomicAdd(&result[threadIdx.x/4], local_sum);  
        __syncthreads();  

        // 只有第一个线程写最终结果  
        if (threadIdx.x % 4 == 0) {  
            C[row] = result[threadIdx.x/4];  
        }  
        __syncthreads();  
    }  
}

void read_numpy_data(float *A, float *B, float *C, int M, int N, int K) {
    
    FILE *fp = fopen("new_A.bin", "rb");
    fread(A, sizeof(float), M * K, fp);
    fclose(fp);
    fp = fopen("new_B.bin", "rb");
    fread(B, sizeof(float), K * N, fp);
    fclose(fp);
    fp = fopen("new_C.bin", "rb");
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
    printf("example value: C_cpu[0][0] = %f, C_cuda[0][0] = %f\n", C_cpu[0], C_cuda[0]);
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
    const int M = 5893, N = 1, K = 1280;
    // const int M = 1536, N = 1, K = 768;
    // const int M = 1, N = 5893, K = 1280;
    float *A_cpu = (float *)malloc(M * K * sizeof(float));
    float *B_cpu = (float *)malloc(K * N * sizeof(float));
    float *C_cpu = (float *)malloc(M * N * sizeof(float));
    float *result_cuda = (float *)malloc(M * N * sizeof(float));
    int threadsPerBlock = 128;
    dim3 DimGrid((M+threadsPerBlock-1)/threadsPerBlock, 1, 1);
    dim3 DimBlock(threadsPerBlock*4, 1, 1);
    read_numpy_data(A_cpu, B_cpu, C_cpu, M, N, K);
    float *cuda_A, *cuda_B, *cuda_C;
    cudaMalloc(&cuda_A, M * K * sizeof(float));
    cudaMalloc(&cuda_B, K * N * sizeof(float));
    cudaMalloc(&cuda_C, M * N * sizeof(float));
    cudaMemcpy(cuda_A, A_cpu, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B, B_cpu, K * N * sizeof(float), cudaMemcpyHostToDevice);
    Sgemm_naive<<<DimGrid, DimBlock>>>(cuda_A, cuda_B, cuda_C, M, N, K);
    cudaMemcpy(result_cuda, cuda_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_result(C_cpu, result_cuda, M, N);
    float sec = testPerformance(Sgemm_naive, DimGrid, DimBlock, M, N, K, 100);
    printf("Kernel time cost: %.6f ms\n", sec);
    sec = testCublasPerformance(M, N, K, 100);
    printf("Cublas time cost: %.6f ms\n", sec);
    cudaError_t err = cudaGetLastError();  
    if (err != cudaSuccess) {  
        printf("CUDA error: %s\n", cudaGetErrorString(err));  
    }
}
// 到时候记得把我们之前照着知乎的那个分块也写一下, 那种就是没充分利用n=1的情况直接套东西,到最后最好的结果是163ms