#include <ATen/cuda/CUDABlas.h>

// A: 1*1280, B: 1280*5893, C: 1*5893
// 现在得改写了，A: 5893 * 1280, B: (1280,), C: (5893,)
__global__ void Sgemm_stable(const float * A,const float * B, float * __restrict__ C, int M, int N, int K) {
    
    const int BLOCK_SIZE_M = 128;// 一个block中处理的A矩阵行数,或者说一个block处理的A矩阵大小为BLOCK_SIZE_M*BLOCK_SIZE_K
    const int BLOCK_SIZE_N = 1;// 一个block中处理的B矩阵列数,或者说一个block处理的B矩阵大小为BLOCK_SIZE_K*BLOCK_SIZE_N
    const int BLOCK_SIZE_K = 8;// 一个block中处理的A矩阵列数和B矩阵行数,或者说一个block处理的C矩阵大小为BLOCK_SIZE_M*BLOCK_SIZE_N
    // 在上面的设置下，需要开启的线程数就是2048*2048/128/128=16*16=256
    // 在矩阵分块的基础上，我们对每一个block中的A和B矩阵继续进行分块，让每个线程处理rm*rn的子矩阵，那么一个block中的线程数就是BLOCK_SIZE_M*BLOCK_SIZE_N/THREAD_SIZE_Y/THREAD_SIZE_X
    const int TM = 4;
    const int TN = 1;
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    // if (bx==0&&by==0&&tid==0) printf("A[0][0] = %f, B[0][0] = %f, A[0][1] = %f, B[0][1] = %f\n", A[0], B[0], A[1], B[1]);
    __shared__ float s_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float s_b[BLOCK_SIZE_K];
    float operator_a[TM];// 在最后计算的时候，将东西load到register，然后和s_a做乘法，实测从162变成158
    float r_c[TM] = {0.0f};
    int load_A_rows = tid >> 5;// 在block中当前线程在B应该load的行
    int load_A_cols = (tid & 31) << 2;// 在block中当前线程在B应该load的列
    // 算了，让所有线程都load A的值吧
    #pragma unroll
    for (int bk = 0; bk < (K+BLOCK_SIZE_K-1)/BLOCK_SIZE_K;++bk) {
        int load_B_cols = bk * BLOCK_SIZE_K;
        // 注意我们只开了32个线程，所以没办法一次就把s_b里的所有东西都load进来
        for (int i=0;i<BLOCK_SIZE_K;++i) {
            if (load_B_cols + i < K) {
                s_b[i] = B[myOFFSET(0, load_B_cols + i, K)];
            } else {
                s_b[i] = 0.0f;
            }
        }
        __syncthreads();
        // 接下来是一个循环将所有的B load进来，但是要注意边界检查
        // 但是这种做贡献的矩阵乘法写法，好像只要在最后数据写回的时候做边界检查就可以了。
        // 这里爆了一个cudaError: Misaligned address错误，原因是我在这里写的时候写到了B的边界外面去了
        // 就是这个循环的问题
        // cnm, 怎么有的float4正常用，有的就报错，服了，直接float吧
        int global_load_a_row_start = load_A_rows + bk * BLOCK_SIZE_K;        
        int global_load_a_col = bx*BLOCK_SIZE_N + load_A_cols;
        // 现在是32个线程，每个线程load1个float,此时load_B_rows = tid/128, load_B_cols = (tid%32)
        // 现在一个row要load4遍
        #pragma unroll
        for (int i=0;i<BLOCK_SIZE_K;++i) {
            int current_load_A_col = tid << 2;
            #pragma unroll
            for (int j=0;j<4;++j) {
                if (global_load_a_row_start + i < M && global_load_a_col + j < K) {
                    s_a[current_load_A_col+j][i] = A[myOFFSET(global_load_a_row_start + i, global_load_a_col + j, K)];
                } else {
                    s_a[current_load_A_col+j][i] = 0.0f;
                }
            }
        }
        __syncthreads();
        // 我试一下load到寄存器里再做乘法会不会更快

        // 但是这里用float4更快
        #pragma unroll
        for (int k=0;k<BLOCK_SIZE_K;++k) {
            // myFETCH_FLOAT4(operator_a) = myFETCH_FLOAT4(s_a[tx*TM][k]);
            #pragma unroll
            for (int n=0;n<TM;++n) {
                r_c[n] += s_a[tx*TM][k] * operator_a[n];
            }
        }
        // __syncthreads();
    }
    // 这个好奇怪啊，用float4反而更慢了
    #pragma unroll
    for (int j=0;j<TM;++j) {
        int global_c_col = bx*BLOCK_SIZE_N + tx*TM+j;
        if (global_c_col < M) {
            C[myOFFSET(0, global_c_col, N)] = r_c[j];
        }
    }
}

void call_my_sgemm(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 DimGrid((N+127)/128, 1, 1);
    dim3 DimBlock(32, 1, 1);
    Sgemm_stable<<<DimGrid, DimBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}