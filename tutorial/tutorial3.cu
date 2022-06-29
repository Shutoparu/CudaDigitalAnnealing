const int BLOCK_SIZE = 5;

// using ILP 2 to improve the performance
__global__ void matrixMulSharedILPkernel(float* A, float* B, float* C, int width){
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val[2] = {0.0f};

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int iter = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for(int i = 0; i < iter; i++){
        // read data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x]=A[row * width+i*BLOCK_SIZE+threadIdx.x];
        shTileA[threadIdx.y+16][threadIdx.x]=A[(row+16)*width+i*BLOCK_SIZE+threadIdx.x];
        shTileB[threadIdx.y][threadIdx.x]=B[(i*BLOCK_SIZE+threadIdx.y)*width+col];
        shTileB[threadIdx.y+16][threadIdx.x]=B[(i*BLOCK_SIZE+threadIdx.y+16)*width+col];
        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++){
            val[0] += shTileA[threadIdx.y][j] * shTileB[j][threadIdx.x];
            val[1] += shTileA[threadIdx.y + 16][j] * shTileB[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * width + col] = val[0];
    C[(row + 16) * width + col] = val[1];
}
