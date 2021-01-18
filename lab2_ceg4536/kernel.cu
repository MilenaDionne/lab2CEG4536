
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 8
#include <iostream>

using namespace std;

__global__ void sum_reduction(int* input)
{
    //CUDA threads --> block can be split into n threads, 
    const int number_of_blocks = threadIdx.x; //block = 1
    int number_of_threads = blockDim.x; //thread = 4, ainsi cette fonction s'exécutera 4 fois 

    auto step_size = 1;
    

    while (number_of_threads > 0)
    {
        if (number_of_blocks < number_of_threads)
        {
            const auto fst = number_of_blocks * step_size * 2;
            const auto snd = fst + step_size;
            input[fst] += input[snd];
        }

        step_size <<= 1;
        number_of_threads >>= 1;
    }
}

void fill_array(int* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = i+1;
    }
}


int main()
{
    //size of memory for 8 elements 
    const int size = N * sizeof(int);

    //Tableau orginal 
    int* a, * result;
    int* d_a;


    //calculate cuda time 
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
    //Allocate memory 
    a = (int*)malloc(size); 
    result = (int*)malloc(size); 
    cudaMalloc((void**)&d_a, size);


    //Initialize table 
    fill_array(a, N);
    
    //Transfert les données de l’hôte à l’appareil
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    //Start time recording 
    cudaEventRecord(start, 0);
    
    //Call kernel on 1 block with 8 thread divided by 2
    sum_reduction << <1, N/2 >> > (d_a);

    
    //cudaDeviceSynchronize(); No need to synchronise kernel in this program. Can stay asynchrone 
    
    //Copy to host 
    cudaMemcpy(result, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cout << "La somme est de " << result[0] << " pour " << N << " elements" << endl;
    cout << "La temps de performance Cuda pour 1 block de 8 thread est: " << time << endl; 

    //Destroy event object start and stop 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //Free card space 
    cudaFree(d_a);

    //Free memory 
    free(a);
    free(result);
    
    return 0;
}

