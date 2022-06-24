#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {

        printf("Cuda Error: %s, %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}


/**
 * @brief randomly choose an index with non-zero value from the given array
 *
 * @param arr input array
 * @param size size of the array
 * @return the index of a random non-zero value from the array
 */
int randChoose(double* arr, int size) {
    int index = rand() % size;

    while (arr[index] == 0) {
        index = rand() % size;
    }
    return index;
}


/**
 * @brief find the minimum value of the given array
 *
 * @param arr input array
 * @param size the size of the array
 * @return return the minimum value of the array
 */
double min(double* arr, int size) {
    double min = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > min) {
            min = arr[i];
        }
    }
    return min;
}


/**
 * @brief sum up the given aray
 *
 * @param arr input array
 * @param size the size of the array
 * @return the sum of the array
 */
double sum(double* arr, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}


/**
 * @brief calculate array dot product
 *
 * @param arr1 first input array
 * @param arr2 second input array
 * @param dim size of the array
 * @return the dot product of the two arrays
 */
double dot(int* arr1, double* arr2, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += arr1[i] * arr2[i];
    }
    return sum;
}


/**
 * @brief calculate the energy with given qubo matrix and binary state
 *
 * @param b array representing binary
 * @param Q qubo matrix
 * @param dim dimention of the array and matrix
 * @return the calculated energy
 */
__global__ void calculateEnergy(int* b, double* Q, double* tempArr, int dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim) {
        tempArr[i] = 0;
        for (int n = 0; n < dim; n++) {
            tempArr[i] += Q[i * dim + n] * b[n];
        }
        tempArr[i] = tempArr[i] * b[i];
    }

}


/**
 * @brief calculate the energy change per bit flip, record the result and return an array of the result
 *
 * @param b the binary array
 * @param Q the qubo matrix
 * @param dim the dimention of the matrix and array
 * @param offset offset if the result is not accepted
 * @param beta a factor to accept randomness
 * @param stat the array to be returned, include [0] acceptance and [1] energy change
 */
__global__ void slipBinary(int* b_copy, double* Q, int dim, double offset, double beta, double* stat, double threshold) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int flipped = 0;
    if (i < dim) {
        // get energy change for flipping the bit [i] (check delta_E)
        if (b_copy[i] != 1) {
            b_copy[i] = 1;
            flipped = 1;
        }

        stat[dim + i] = 0;

        for (int n = 0; n < dim; n++) {
            stat[dim + i] += b_copy[n] * Q[i * dim + n];
        }

        if (flipped != 0) {
            stat[dim + i] = 2 * stat[dim + i] - offset;
        } else {
            stat[dim + i] = -2 * stat[dim + i] - offset;
        }

        // check energy or check % (check pass)
        double p = exp(-stat[dim + i] * beta);
        if (stat[dim + i] < 0) {
            stat[i] = 1;
        } else if (p > threshold) {
            stat[i] = 1;
        } else {
            stat[i] = 0;
        }
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
void getAnnealingBeta(int betaStart, int betaStop, double* beta, int sweeps) {

    double logBetaStart = log((double)betaStart);
    double logBetaStop = log((double)betaStop);
    double logBetaRange = (logBetaStop - logBetaStart) / (double)sweeps;
    for (int i = 0; i < sweeps; i++) {

        beta[i] = exp(logBetaStart + logBetaRange * i);
    }
}

/**
 * @brief the function that runs the digital annealing algorithm
 *
 * @param b binary array
 * @param Q qubo matrix
 * @param energy energy matrix to be returned, will record energy after per flip
 * @param dim dimention of binary array and qubo matrix
 * @param sweeps number of iterations to be done
 */
void digitalAnnealing(int* b, double* Q, double* energy, int dim, int sweeps) {

    int blocks = 32 * 8;
    int threads = dim / blocks + 1;

    int betaStart = 1;
    int betaStop = 50;

    double* beta;
    beta = (double*)malloc(sweeps * sizeof(double));
    getAnnealingBeta(betaStart, betaStop, beta, sweeps);

    double offset = 0;
    double offsetIncreasingRate = 0.1;

    double* stat;
    cudaMalloc(&stat, 2 * dim * sizeof(double));

    double* stat_host;
    cudaMallocHost(&stat_host, 2 * dim * sizeof(double));

    int* b_copy;
    cudaMalloc(&b_copy, dim * sizeof(int));

    double* Q_copy;
    cudaMalloc(&Q_copy, dim * dim * sizeof(double));
    cudaMemcpy(Q_copy, Q, dim * dim * sizeof(double), cudaMemcpyHostToDevice);

    double* tempArr;
    cudaMalloc(&tempArr, dim * sizeof(double));

    double* tempArr_Host;
    cudaMallocHost(&tempArr_Host, dim * sizeof(double));

    cudaMemcpy(b_copy, b, dim * sizeof(int), cudaMemcpyHostToDevice);

    for (int n = 0; n < sweeps; n++) {

        slipBinary << < blocks, threads >> > (b_copy, Q_copy, dim, offset, beta[n], stat, rand() / (double)RAND_MAX);
        cudaDeviceSynchronize();
        cudaMemcpy(stat_host, stat, 2 * dim * sizeof(double), cudaMemcpyDeviceToHost);

        // stat[0] = accept, stat[1] = delta_E
        if (sum(stat_host, dim) == 0) {
            offset += offsetIncreasingRate * min(&stat_host[dim], dim);
        } else {
            int index = randChoose(stat_host, dim);
            b[index] = b[index] * -1 + 1;
            offset = 0;
        }

        cudaMemcpy(b_copy, b, dim * sizeof(int), cudaMemcpyHostToDevice);
        calculateEnergy << <blocks, threads >> > (b_copy, Q_copy, tempArr, dim);
        cudaDeviceSynchronize();
        cudaMemcpy(tempArr_Host, tempArr, dim*sizeof(double), cudaMemcpyDeviceToHost);
        energy[n] = sum(tempArr_Host,dim);
    }
    free(beta);
    cudaFree(stat);
    cudaFreeHost(stat_host);
    cudaFree(b_copy);
    cudaFree(Q_copy);
    cudaFree(tempArr);
    cudaFreeHost(tempArr_Host);
}

int main() {

    srand(1);

    // create a random 40 * 40 array Q
    // create an inital state([1]) bit array b
    int dim = 1500;
    double* Q;
    int* b;
    cudaMallocHost(&Q, dim * dim * sizeof(double));
    cudaMallocHost(&b, dim * sizeof(int));
    for (int i = 0; i < dim; i++) {
        b[i] = 1;
    }
    for (int i = 0; i < dim * dim; i++) {
        Q[i] = rand() / ((double)(RAND_MAX - 1) / 2 + 1) - 1;
    }
    int sweeps = 100000;
    double* energy;
    cudaMallocHost(&energy, sweeps * sizeof(double));


    clock_t begin = clock();
    digitalAnnealing(b, Q, energy, dim, sweeps);
    clock_t end = clock();

    double time = (double)(end - begin) / CLOCKS_PER_SEC;


    printf("time=%.5f sec\n", time);

    cudaFree(Q);
    cudaFree(b);
    cudaFree(energy);
    return 0;
}
