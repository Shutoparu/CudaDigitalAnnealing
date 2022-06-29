#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

/**
 * @brief used to check if cuda code goes wrong
 */
void checkCudaError()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {

        printf("Cuda Error: %s, %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

/**
 * @brief sum up the given aray
 *
 * @param arr input array
 * @param size the size of the array
 * @return the sum of the array
 */
double sum(double *arr, int size)
{
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += arr[i];
    }
    return sum;
}

/**
 * @param arr input binary array
 * @param size size of the array
 * @return the index of a random non-zero value from the array
 */
int randChoose(double *arr, int size)
{

    int nonZeroNum = 0;

    int *indicies;
    indicies = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        if (arr[i] != 0)
        {
            indicies[nonZeroNum] = i;
            nonZeroNum++;
        }
    }

    int index = indicies[rand() % nonZeroNum];
    free(indicies);

    return index;
}

/**
 * @brief find the minimum value of the given array
 *
 * @param arr input array
 * @param size the size of the array
 * @return return the minimum value of the array
 */
double min(double *arr, int size)
{
    double min = arr[0];
    for (int i = 1; i < size; i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }
    return min;
}

/**
 * @brief calculate the energy with given qubo matrix and binary state
 *
 * @param b array representing binary
 * @param Q qubo matrix
 * @param tempArr a temporary array to store the dot product of b^T * (Q*b)
 * @param dim dimention of the array and matrix
 */
__global__ void calculateEnergy(int *b, double *Q, double *tempArr, int dim)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim)
    {
        tempArr[i] = 0;
        for (int n = 0; n < dim; n++)
        {
            tempArr[i] += Q[i * dim + n] * b[n];
        }
        tempArr[i] = tempArr[i] * b[i];
    }
}

/**
 * @brief calculate the energy change per bit flip, record the result and return an array of the result
 *
 * @param b_copy the binary array
 * @param Q the qubo matrix
 * @param dim the dimention of the matrix and array
 * @param offset constant to deduct if the result was not accepted in the previous round
 * @param beta a factor to accept randomness
 * @param stat the array to be returned, include [0] acceptance and [1] energy change
 * @param seed a seed to create random double between (0,1] in kernel
 */
__global__ void slipBinary(int *b_copy, double *Q, int dim, double offset, double beta, double *stat, double seed)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim)
    {
        int flipped = 0;
        double delta_E;
        curandState state;
        curand_init(seed, i, 0, &state);

        // get energy change for flipping the bit [i] (check delta_E)
        if (b_copy[i] == 0)
        {
            flipped = 1;
        }

        for (int n = 0; n < dim; n++)
        {
            if (n == i && flipped == 1)
            {
                delta_E += Q[i * dim + n]; // time consuming
            }
            else
            {
                delta_E += b_copy[n] * Q[i * dim + n]; // time consuming
            }
        }

        if (flipped != 0)
        {
            delta_E = 2 * delta_E - offset;
        }
        else
        {
            delta_E = -2 * delta_E - offset;
        }

        // check energy or check % (check pass)
        double p = exp(-delta_E * beta);
        if (delta_E < 0)
        {
            stat[i] = 1;
        }
        else if (p > curand_uniform_double(&state))
        {
            stat[i] = 1;
        }
        else
        {
            stat[i] = 0;
        }
        stat[dim + i] = delta_E;
    }
}

/**
 * @brief create the beta array
 *
 * @param betaStart starting value of beta
 * @param betaStop ending value of beta
 * @param beta the beta array to be returned
 * @param sweeps the length of beta array
 */
void getAnnealingBeta(int betaStart, int betaStop, double *beta, int sweeps)
{

    double logBetaStart = log((double)betaStart);
    double logBetaStop = log((double)betaStop);
    double logBetaRange = (logBetaStop - logBetaStart) / (double)sweeps;
    for (int i = 0; i < sweeps; i++)
    {

        beta[i] = exp(logBetaStart + logBetaRange * i);
    }
}

/**
 * @brief the function that runs the digital annealing algorithm
 *
 * @param b binary array
 * @param Q qubo matrix
 * @param dim dimention of binary array and qubo matrix
 * @param energy energy matrix to be returned, will record energy after per flip
 * @param sweeps number of iterations to be done
 */
void digitalAnnealing(int *b, double *Q, int dim, double *energy, int sweeps)
{

    int blocks = 32 * 8;
    int threads = dim / blocks + 1;

    int betaStart = 1;
    int betaStop = 50;

    double *beta;
    beta = (double *)malloc(sweeps * sizeof(double));
    getAnnealingBeta(betaStart, betaStop, beta, sweeps);

    double offset = 0;
    double offsetIncreasingRate = 0.1;

    double *stat;
    cudaMalloc(&stat, 2 * dim * sizeof(double));

    double *stat_host;
    cudaMallocHost(&stat_host, 2 * dim * sizeof(double));

    int *b_copy;
    cudaMalloc(&b_copy, dim * sizeof(int));

    double *Q_copy;
    cudaMalloc(&Q_copy, dim * dim * sizeof(double));
    cudaMemcpy(Q_copy, Q, dim * dim * sizeof(double), cudaMemcpyHostToDevice);

    // for calculating energy
    double *tempArr;
    cudaMalloc(&tempArr, dim * sizeof(double));

    // for calculating energy
    double *tempArr_Host;
    cudaMallocHost(&tempArr_Host, dim * sizeof(double));

    for (int n = 0; n < sweeps; n++)
    {

        cudaMemcpy(b_copy, b, dim * sizeof(int), cudaMemcpyHostToDevice);

        slipBinary<<<blocks, threads>>>(b_copy, Q_copy, dim, offset, beta[n], stat, (double)rand());
        cudaDeviceSynchronize();
        cudaMemcpy(stat_host, stat, 2 * dim * sizeof(double), cudaMemcpyDeviceToHost);

        // stat[0] = accept, stat[1] = delta_E
        if (sum(stat_host, dim) == 0)
        {
            offset += offsetIncreasingRate * min(&stat_host[dim], dim);
        }
        else
        {
            int index = randChoose(stat_host, dim);
            b[index] = b[index] * -1 + 1;
            offset = 0;
        }

        // calculate energy ; only needed for testing
        {
            cudaMemcpy(b_copy, b, dim * sizeof(int), cudaMemcpyHostToDevice);
            calculateEnergy<<<blocks, threads>>>(b_copy, Q_copy, tempArr, dim);
            cudaDeviceSynchronize();
            cudaMemcpy(tempArr_Host, tempArr, dim * sizeof(double), cudaMemcpyDeviceToHost);
            energy[n] = sum(tempArr_Host, dim);
        }
    }
    free(beta);
    cudaFree(stat);
    cudaFreeHost(stat_host);
    cudaFree(b_copy);
    cudaFree(Q_copy);
    cudaFree(tempArr);
    cudaFreeHost(tempArr_Host);
}

int main()
{

    int dim = 1500;

    // create a random 40 * 40 array Q
    // create an inital state([1]) bit array b
    srand(1);
    double *Q;
    int *b;
    cudaMallocHost(&Q, dim * dim * sizeof(double));
    cudaMallocHost(&b, dim * sizeof(int));
    for (int i = 0; i < dim; i++)
    {
        b[i] = 1;
    }
    for (int i = 0; i < dim * dim; i++)
    {
        Q[i] = rand() / ((double)(RAND_MAX - 1) / 2 + 1) - 1;
    }

    int sweeps = 100;
    double *energy;
    cudaMallocHost(&energy, sweeps * sizeof(double));

    digitalAnnealing(b, Q, dim, energy, sweeps);

    int stride = 10;
    for (int i = 0; i < sweeps / stride; i++)
    {
        printf("i=%d --> e=%.5f\n", i * stride, energy[i * stride]);
    }

    cudaFree(Q);
    cudaFree(b);
    cudaFree(energy);
    return 0;
}

/////////////////////////////////////////////////////////////////////////
/// Below is the code that Python code calls to execute the algorithm ///
/////////////////////////////////////////////////////////////////////////

extern "C"
{
    double digitalAnnealingPy(int *b, double *Q, int dim, int sweeps);
}

const int STRIDE = 30;

__global__ void slipBinaryPy(int *b_copy, double *Q, int dim, double offset, double beta, double *stat, double seed)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim)
    {
        int flipped = 0;
        double delta_E;
        curandState state;
        curand_init(seed, i, 0, &state);

        // get energy change for flipping the bit [i] (check delta_E)
        if (b_copy[i] == 0)
        {
            flipped = 1;
        }

        __shared__ int sb[STRIDE];

        for (int a = 0; a < dim / STRIDE; a++)
        {

            sb[i % STRIDE] = b_copy[a * STRIDE + i % STRIDE];

            __syncthreads();

            for (int b = 0; b < STRIDE; b++)
            {
                if (a * STRIDE + b == i && flipped == 1)
                {
                    delta_E += Q[i * dim + a * STRIDE + b];
                }
                else
                {
                    delta_E += sb[b] * Q[i * dim + a * STRIDE + b];
                }
            }
            __syncthreads();
        }
        __syncthreads();

        if (flipped != 0)
        {
            delta_E = 2 * delta_E - offset;
        }
        else
        {
            delta_E = -2 * delta_E - offset;
        }

        // check energy or check % (check pass)
        double p = exp(-delta_E * beta);
        if (delta_E < 0)
        {
            stat[i] = 1;
        }
        else if (p > curand_uniform_double(&state))
        {
            stat[i] = 1;
        }
        else
        {
            stat[i] = 0;
        }
        stat[dim + i] = delta_E;
    }
}
/**
 * @brief the function that runs the digital annealing algorithm
 *
 * @param b binary array
 * @param Q qubo matrix
 * @param dim dimention of binary array and qubo matrix
 * @param sweeps number of iterations to be done
 */
double digitalAnnealingPy(int *b, double *Q, int dim, int sweeps)
{

    // srand(time(NULL));
    srand(1);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // int blocks = prop.multiProcessorCount * 4;
    int threads = STRIDE;
    int blocks = dim / threads + 1;

    int betaStart = 1;
    int betaStop = 50;

    double *beta;
    beta = (double *)malloc(sweeps * sizeof(double));
    getAnnealingBeta(betaStart, betaStop, beta, sweeps);

    double offset = 0;
    double offsetIncreasingRate = 0.1;

    double *stat;
    cudaMalloc(&stat, 2 * dim * sizeof(double));

    double *stat_host;
    cudaMallocHost(&stat_host, 2 * dim * sizeof(double));

    int *b_copy;
    cudaMalloc(&b_copy, dim * sizeof(int));

    double *Q_copy;
    cudaMalloc(&Q_copy, dim * dim * sizeof(double));
    cudaMemcpy(Q_copy, Q, dim * dim * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(b_copy, b, dim * sizeof(int), cudaMemcpyHostToDevice);

    for (int n = 0; n < sweeps; n++)
    {

        slipBinaryPy<<<blocks, threads>>>(b_copy, Q_copy, dim, offset, beta[n], stat, (double)rand());
        cudaDeviceSynchronize();
        cudaMemcpy(stat_host, stat, 2 * dim * sizeof(double), cudaMemcpyDeviceToHost);

        // stat[0] = accept, stat[1] = delta_E
        if (sum(stat_host, dim) == 0)
        {
            offset += offsetIncreasingRate * min(&stat_host[dim], dim);
        }
        else
        {
            int index = randChoose(stat_host, dim);
            b[index] = b[index] * -1 + 1;
            offset = 0;
            cudaMemcpy(b_copy, b, dim * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    // calculate energy ; only needed for testing

    // for calculating energy
    double *tempArr;
    cudaMalloc(&tempArr, dim * sizeof(double));

    // for calculating energy
    double *tempArr_Host;
    cudaMallocHost(&tempArr_Host, dim * sizeof(double));

    double energy = 0;
    {
        cudaMemcpy(b_copy, b, dim * sizeof(int), cudaMemcpyHostToDevice);
        calculateEnergy<<<blocks, threads>>>(b_copy, Q_copy, tempArr, dim);
        cudaDeviceSynchronize();
        cudaMemcpy(tempArr_Host, tempArr, dim * sizeof(double), cudaMemcpyDeviceToHost);
        energy = sum(tempArr_Host, dim);
    }

    free(beta);
    cudaFree(stat);
    cudaFreeHost(stat_host);
    cudaFree(b_copy);
    cudaFree(Q_copy);
    cudaFree(tempArr);
    cudaFreeHost(tempArr_Host);

    return energy;
}