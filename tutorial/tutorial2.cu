const int BLOCK_SIZE = 5;
//核函数的具体实现
__global__ void matmul_ShareMemory(int *M,int *N,int *P,int width){
    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * BLOCK_SIZE + tx;
    int Row = by * BLOCK_SIZE + ty;

    int Pervalue = 0;
    //有多少个BLOCK_SIZE，每个循环计算一个块的大小
    for(int i = 0;i < width / BLOCK_SIZE;i++){
        Mds[ty][tx] = M[Row * width + (i * BLOCK_SIZE + tx)];
        Nds[ty][tx] = N[Col + (i * BLOCK_SIZE + ty) * width];
        __syncthreads();

        //BLOCK_SIZE相乘
        for(int k = 0;k < BLOCK_SIZE;k++)
            Pervalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    P[Row * width + Col] = Pervalue;
}
