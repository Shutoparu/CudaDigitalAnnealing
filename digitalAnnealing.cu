#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/**
 * randomly choose an index with non-zero value from the given array
 *
 * arr: input array
 * size: size of the array
 *
 * return the index of a random non-zero value from the array
 */
int randChoose(double* arr, int size){
    int index = rand()%size;

    while(arr[index] == 0){
        index = rand()%size;
    }
    return index;
}


/**
 * find the minimum value of the given array
 * 
 * arr: input array
 * size: the size of the array
 * 
 * return the minimum value of the array
 */
double min(double* arr, int size){
    double min = arr[0];
    for(int i=1; i<size; i++){
        if(arr[i]>min){min=arr[i];}
    }
    return min;
}


/**
 * sum upp the given array
 * 
 * arr: input array
 * size: the size of the array
 *
 * return the sum of the array
 */
double sum(double* arr, int size){
    double sum = 0;
    for(int i=0; i<size; i++){sum+=arr[i];}
    return sum;
}


/**
 * claculate array index by index multiplication and sum up, as I'm not sure if c has a standard library for it
 * 
 * arr1: first input array
 * arr2: second input array
 * dim_i size of the array
 * 
 * return the sum of multiplied array
 */
double pos_mul(int* arr1, double* arr2, int dim_i){
    double sum = 0;
    for(int i=0; i<dim_i; i++){
        sum += arr1[i] * arr2[i];
    }
    return sum;
}


/**
 * calculate the energy with given qubo matrix and binary state
 *
 * b: array representing binary
 * Q: qubo matrix
 * dim: dimention of the array and matrix
 *
 * return the calculated energy
 */
double calculateEnergy(int* b, double* Q, int dim){
    double* tmparr;
    tmparr = (double*)malloc(dim*sizeof(double));
    
    for(int i=0; i<dim; i++){
       
        tmparr[i] = pos_mul(b,&Q[i*dim],dim); 
    }
    double energy = pos_mul(b,tmparr,dim);
    
    free(tmparr);    
    return energy;
}


/**
 * calculate the energy change per bit flip, record the result and return an array of the result
 * 
 * b: the binary array
 * Q: the qubo matrix
 * dim_i: the first dimention of the qubo matrix, also treated as n in the og python file
 * dim_j: the second dimention of the qubo matrix
 * offset
 * beta
 * stat: the array to be returned, include [0] if we are accepting the change and [1] the energy change
 */
void slipBinary(int* b, double* Q, int dim, double offset, double beta, double** stat){

    // copy b
    int* b_copy;
    b_copy = (int*)malloc(dim*sizeof(int));
    memcpy(b_copy, b, dim*sizeof(int));

    double delta_E = 0;    

    for(int i=0; i<dim; i++){

        // get energy change for flipping the bit [i] (check delta_E)
        if (b_copy[i]==1){
            delta_E = (double)(-2 * pos_mul(b_copy, &Q[i*dim], dim)) - offset;
        }else{
            b_copy[i] = 1;
            delta_E = (double)(2 * pos_mul(b_copy, &Q[i*dim], dim)) - offset;
        }

        // check energy or check % (check pass)
        int pass = 0;
        double p = exp(-delta_E * beta);
        if(delta_E < 0){
            pass = 1;
        }else if(p > rand() / (double) RAND_MAX){
            pass = 1;
        }else{
            pass = 0;
        }

        // save (pass,delta_E) to output array
        stat[0][i] = pass;
        stat[1][i] = delta_E;
    }
}


/**
 * create the beta array
 * 
 * betaStart / betaStop : the starting and ending value of beta
 * beta : input an empty double array. output the beta array
 * sweeps : the length of beta array
 */
void getAnnealingBeta(int betaStart, int betaStop, double* beta, int sweeps){

    double logBetaStart = log((double)betaStart);
    double logBetaStop = log((double)betaStop);
    double logBetaRange = (logBetaStop - logBetaStart)/(double)sweeps;
    for(int i=0; i<sweeps; i++){
        
        beta[i] = exp(logBetaStart + logBetaRange * i);
    }
}


/**
 */
void digitalAnnealingMultiThread(int* b, double* Q, double* energy, int dim, int sweeps){
    
    int betaStart = 1;
    int betaStop = 50;
    
    double* beta;
    beta = (double*)malloc(sweeps*sizeof(double));
    getAnnealingBeta(betaStart,betaStop,beta,sweeps);

    double offset = 0;
    double offsetIncreasingRate = 0.1;

    double** stat;
    stat = (double**)malloc(2*sizeof(double*));
    for(int i=0; i<2; i++){stat[i] = (double*)malloc(dim*sizeof(double));}

    double* accept;
    accept = (double*)malloc(dim*sizeof(double));
    int index;

    for(int n=0; n<sweeps; n++){
    
        slipBinary(b, Q, dim, offset, beta[n], stat);

        // stat[0] = accept, stat[1] = delta_E
        
        memcpy(accept, stat[0], dim*sizeof(double));
        
        if(sum(accept,dim) == 0){
            offset += offsetIncreasingRate * min(stat[1],dim);
        }else{
            index = randChoose(accept,dim);
            b[index] = b[index] * -1 +1;
            offset = 0;
        }

        energy[n] = calculateEnergy(b,Q,dim);
    }
    
    free(beta);
    for(int i=0; i<3; i++){free(stat[i]);}
    free(stat);
    free(accept);
}


int main(){

    srand( 1 );

    // create a random 40 * 40 array Q
    // create an inital state([1]) bit array b
    int dim = 2000;
    double* Q;
    int* b;
    Q = (double *)malloc(dim * dim * sizeof(double));
    b = (int *)malloc(dim * sizeof(int));
    for (int i=0; i<dim; i++){
        b[i] = 1;
    }
    for (int i=0; i<dim*dim; i++){
        Q[i] = rand()/((double)(RAND_MAX-1)/2+1)-1;
    }

    int sweeps = 1000;
    double* energy;
    energy = (double*)malloc(sweeps*sizeof(double));
    
    clock_t begin = clock();
    digitalAnnealingMultiThread(b,Q,energy,dim,sweeps);  
    clock_t end = clock();

    double time = (double)(end-begin)/CLOCKS_PER_SEC;
    
    int stride = 100;

    printf("time=%.5f sec\n",time);
    for(int i=0; i<sweeps/stride; i++){
        printf("i=%d --> e=%.9f\n",i*stride,energy[i*stride]);
    }
    
    free(Q);
    free(b);
    free(energy);
    return 0;
}
