#include "custom_matmul.h"
__global__ void matmul_tiling(float *A, float *B, float *C, int M, int N, int K) {

    const int TILE_WIDTH = 32;
    // use shared memory to accelerate the computation.
    __shared__ float sub_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float pvalue = 0;
    for (int n=0;n<(N+TILE_WIDTH-1)/TILE_WIDTH;++n) {
        if (row < M && n*TILE_WIDTH + tx < N) {
            sub_A[ty][tx] = A[row*N+n*TILE_WIDTH + tx];
        } else {
            sub_A[ty][tx] = 0;
        }
        if (col < K && n*TILE_WIDTH + ty < N) {
            sub_B[ty][tx] = B[(n*TILE_WIDTH + ty)*K + col];
        } else {
            sub_B[ty][tx] = 0;
        }
        __syncthreads();
        for (int k=0;k<TILE_WIDTH;k+=4) {
            pvalue += sub_A[ty][k] * sub_B[k][tx];
            pvalue += sub_A[ty][k+1] * sub_B[k+1][tx];
            pvalue += sub_A[ty][k+2] * sub_B[k+2][tx];
            pvalue += sub_A[ty][k+3] * sub_B[k+3][tx];
        }
        __syncthreads();
    }
    if (row < M && col < K) {
        C[row*K+col] = pvalue;
    }

}

void launch_matmul(float *A, float *B, float *C, int M, int N, int K) {
    dim3 grid((K+31)/32, (M+31)/32);
    dim3 block(32, 32);
    matmul_tiling<<<grid, block>>>(A, B, C, M, N, K);
}