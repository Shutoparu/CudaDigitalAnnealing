//核函数的具体实现
__global__ void matMul_GlobalKernel(int *A,int *B,int *C,int width){
   int bx = blockIdx.x;
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int Col = bx * blockDim.x + tx;
   int Row = by * blockDim.y + ty;

   int perValue = 0;
   for(int i = 0; i < width; i++){
       perValue += A[Row * width + i] * B[i * width + Col];
   }
   C[Row * width + Col] = perValue;
}
