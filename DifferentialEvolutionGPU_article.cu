/* Copyright 2017 Ian Rankin
*
* Permission is hereby granted, free of charge, to any person obtaining a copy of this
* software and associated documentation files (the "Software"), to deal in the Software
* without restriction, including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
* to whom the Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or
* substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// DifferentialEvolutionGPU.cu
// This file holds the GPU kernel functions required to run differential evolution.
// The software in this files is based on the paper:
// Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continous Spaces,
// Rainer Storn, Kenneth Price (1996)
//
// But is extended upon for use with GPU's for faster computation times.
// This has been done previously in the paper:
// Differential evolution algorithm on the GPU with C-CUDA
// Lucas de P. Veronese, Renato A. Krohling (2010)
// However this implementation is only vaguly based on their implementation.
// Translation: I saw that the paper existed, and figured that they probably
// implemented the code in a similar way to how I was going to implement it.
// Brief read-through seemed to be the same way.
//
// The paralization in this software is done by using multiple cuda threads for each
// agent in the algorithm. If using smaller population sizes, (4 - 31) this will probably
// not give significant if any performance gains. However large population sizes are more
// likly to give performance gains.
//
// HOW TO USE:
// To implement a new cost function write the cost function in DifferentialEvolutionGPU.cu with the header
// __device float fooCost(const float *vec, const void *args)
// @param vec - sample parameters for the cost function to give a score on.
// @param args - any set of arguements that can be passed at the minimization stage
// NOTE: args any memory given to the function must already be in device memory.
//
// Go to the header and add a specifier for your cost functiona and change the COST_SELECTOR
// to that specifier. (please increment from previous number)
//
// Once you have a cost function find the costFunc function, and add into
// preprocessor directives switch statement
//
// ...
// #elif COST_SELECTOR == YOUR_COST_FUNCTION_SPECIFIER
//      return yourCostFunctionName(vec, args);
// ...
//


#include <curand_kernel.h>


#include <cuda_runtime.h>
// for random numbers in a kernel
#include "DifferentialEvolutionGPU.h"

// for FLT_MAX
#include <cfloat>

#include <iostream>

// for clock()
#include <ctime>
#include <cmath>

// basic function for exiting code on CUDA errors.
// Does no special error handling, just exits the program if it finds any errors and gives an error message.
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        // TODO: Revert
//        exit(code);
    }
}

#define PI 3.14159f

// -----------------IMPORTANT----------------
// costFunc - this function must implement whatever cost function
// is being minimized.
// Feel free to delete all code in here.
// This is a bit of a hack and not elegant at all. The issue is that
// CUDA doesn't support function passing device code between host
// software. There is a possibilty of using virtual functions, but
// was concerned that the polymorphic function have a lot of overhead
// So instead use this super ugly method for changing the cost function.
//
// @param vec - the vector to be evaulated.
// @param args - a set of user arguments.
// @param dim - number of dimensions.

// Functions:
// Shifted Rastrigin’s Function
__host__ __device__ float rastriginFunc(const float *vec, const void *args, const int dim)
{
    float res = 10 * dim;

    float x;
    for (int i = 0; i < dim; i++) {
        x = vec[i];
        res += x * x + 10 * cos(2 * PI * x);
    }

    return res;
}

// Shifted Rosenbrock’s Function
__host__ __device__ float rosenblockFunc(const float *vec, const void *args, const int dim)
{
    float res = 0;
    float curr, next;
    for (int i = 0; i < dim - 1; i++) {
        curr = vec[i], next = vec[i + 1];
        res += 100 * pow(next - curr * curr, 2) + (curr - 1) * (curr - 1);
    }

    return res;
}

// Shifted Griewank’s function
__host__ __device__ float griewankFunc(const float *vec, const void *args, const int dim)
{
    float a = 1, b = 1; 
    
    float x;
    for (int i = 0; i < dim; i++) {
        x = vec[i];

        a += (x * x);
        b *= cos(x / sqrt(i + 1));
    }

    return a / 4000 - b;
}

// Shifted Sphere’s Function
__host__ __device__ float sphereFunc(const float *vec, const void *args, const int dim)
{
    float res = 0;

    for (int i = 0; i < dim; i++) {
        res += vec[i] * vec[i];
    }

    return res;
}




// costFunc
// This is a selector of the functions.
// Although this code is great for usabilty, by using the preprocessor directives
// for selecting the cost function to use this gives no loss in performance
// wheras a switch statement or function pointer would require extra instructions.
// also function pointers in CUDA are complex to work with, and particulary with the
// architecture used where a standard C++ class is used to wrap the CUDA kernels and
// handle most of the memory mangement used.
__host__ __device__ float costFunc(const float *vec, const void *args, const int dim) {
#if COST_SELECTOR == COST_RASTRIGIN
    return rastriginFunc(vec, args, dim);
#elif COST_SELECTOR == COST_ROSENBROCK
    return rosenblockFunc(vec, args, dim);
#elif COST_SELECTOR == COST_GRIEWANK
    return griewankFunc(vec, args, dim);
#elif COST_SELECTOR == COST_SPHERE
    return sphereFunc(vec, args, dim);
#else
#error Bad cost_selector given to costFunc in DifferentialEvolution function: costFunc
#endif
}

// Mutation indices
#if MUTATION_PARAMS == MUTATION_PARAMS_1
    #define MUTATION_INDICES_COUNT 3
#else
    #define MUTATION_INDICES_COUNT 5
#endif

template <typename T>
void printCudaVector(T *d_vec, int size)
{
    T *h_vec = new T[size];
    gpuErrorCheck(cudaMemcpy(h_vec, d_vec, sizeof(T) * size, cudaMemcpyDeviceToHost));

    std::cout << "{";
    for (int i = 0; i < size; i++) {
        std::cout << h_vec[i] << ", ";
    }
    std::cout << "}" << std::endl;

    delete[] h_vec;
}

__global__ void generateRandomVectorAndInit(float *d_x, float *d_min, float *d_max,
            float *d_cost, void *costArgs, curandState_t *randStates,
            int popSize, int dim, unsigned long seed)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return;

    curandState_t *state = &randStates[idx];
    curand_init(seed, idx,0,state);
    for (int i = 0; i < dim; i++) {
        d_x[(idx*dim) + i] = (curand_uniform(state) * (d_max[i] - d_min[i])) + d_min[i];
    }

    d_cost[idx] = costFunc(&d_x[idx*dim], costArgs, dim);
}

/*
 * Generates 3 non-equal indices for usage in the mutation
 * @param popSize - the population size
 * @param randStates - an array of random number generator states. Array created using createRandNumGen function
 * @param output - a device array used for output
 */
__global__ void generateMutationIndices(
        int popSize,
        curandState_t *randStates,
        int* output
) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    curandState_t *state = &randStates[idx];

    int a;
    int b;
    int c;
    //////////////////// Random index mutation generation //////////////////
    // select a different random number then index
    do { a = curand(state) % popSize; } while (a == idx);
    do { b = curand(state) % popSize; } while (b == idx || b == a);
    do { c = curand(state) % popSize; } while (c == idx || c == a || c == b);

    #if MUTATION_PARAMS != MUTATION_PARAMS_1
        int d, f;
        do { d = curand(state) % popSize; } while (d == idx || d == a || d == b || d == c);
        do { f = curand(state) % popSize; } while (f == idx || f == a || f == b || f == c || f == d);
    #endif

    output[idx * MUTATION_INDICES_COUNT] = a;
    output[idx * MUTATION_INDICES_COUNT + 1] = b;
    output[idx * MUTATION_INDICES_COUNT + 2] = c;

    #if MUTATION_PARAMS != MUTATION_PARAMS_1
        output[idx * MUTATION_INDICES_COUNT + 3] = d;
        output[idx * MUTATION_INDICES_COUNT + 4] = f;
    #endif
}

__global__ void findBest(float* population, float* cost, int popSize, int dim, float* output) {
    extern __shared__ float m_costs[];
    extern __shared__ int m_best_indices[];

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= popSize) {
        return;
    }

    // Populate shared memory
    m_costs[idx] = cost[idx];
    m_best_indices[idx] = idx;

    __syncthreads();

    int t_1, t_2, t_best = 0;
    for (int i = popSize; i > 1; i = (i % 2) + (i / 2)) {
        t_1 = 2 * idx;
        t_2 = 2 * idx + 1;
        
        if (t_2 >= i) {
            if (t_1 >= i) {
                continue;
            } else {
                t_best = t_1;
            }
        } else {
            t_best = m_costs[m_best_indices[t_1]] <= m_costs[m_best_indices[t_2]] ? t_1 : t_2;
        }

        __syncthreads();

        m_best_indices[idx] = m_best_indices[t_best];

        __syncthreads();
    }

    if (idx == 0) {
        int bestIndex = m_best_indices[0];
        for (int i = 0; i < dim; i++) {
            output[i] = population[(bestIndex * dim) + i];
        }
    }
}

// This function handles the entire differentialEvolution, and calls the needed kernel functions.
// @param d_target - a device array with the current agents parameters (requires array with size popSize*dim)
// @param d_best - a device array with the current best element
// @param d_trial - a device array with size popSize*dim (worthless outside of function)
// @param d_cost - a device array with the costs of the last generation afterwards size: popSize
// @param d_target2 - a device array with size popSize*dim (worthless outside of function)
// @param mutationIndices - a device array with indices for mutation
// @param d_min - a list of the minimum values for the set of parameters (size = dim)
// @param d_max - a list of the maximum values for the set of parameters (size = dim)
// @param randStates - an array of random number generator states. Array created using createRandNumGen function
// @param dim - the number of dimensions the equation being minimized has.
// @param popSize - this the population size for DE, or otherwise the number of agents that DE will use. (see DE paper for more info)
// @param CR - Crossover Constant used by DE (see DE paper for more info)
// @param F - the scaling factor used by DE (see DE paper for more info)
// @param costArgs - this a set of any arguments needed to be passed to the cost function. (must be in device memory already)
__global__ void evolutionKernel(float *d_target,
                                float *d_best,
                                float *d_trial,
                                float *d_cost,
                                float *d_target2,
                                int *mutationIndices,
                                float *d_min,
                                float *d_max,
                                curandState_t *randStates,
                                int dim,
                                int popSize,
                                int CR, // Must be given as value between [0,999]
                                float F,
                                void *costArgs)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= popSize) return; // stop executing this block if
                                // all populations have been used
    
    curandState_t *state = &randStates[idx];

#if MUTATION_POINT == MUTATION_POINT_BEST
    #define MUTATION_POINT_ATTR(i) d_best[i]
#else    
    int a = mutationIndices[idx * MUTATION_INDICES_COUNT];
    #define MUTATION_POINT_ATTR(i) d_target[(a*dim) + i]
#endif

#if MUTATION_PARAMS == MUTATION_PARAMS_1
    int x1 = mutationIndices[idx * MUTATION_INDICES_COUNT + 1];
    int x2 = mutationIndices[idx * MUTATION_INDICES_COUNT + 2];

    #define MUTATE() d_trial[(idx*dim)+k] = MUTATION_POINT_ATTR(k) + (F * (d_target[(x1*dim)+k] - d_target[(x2*dim)+k]));
#else
    int x1 = mutationIndices[idx * MUTATION_INDICES_COUNT + 1];
    int x2 = mutationIndices[idx * MUTATION_INDICES_COUNT + 2];
    int x3 = mutationIndices[idx * MUTATION_INDICES_COUNT + 3];
    int x4 = mutationIndices[idx * MUTATION_INDICES_COUNT + 4];

    #define MUTATE() d_trial[(idx*dim)+k] = MUTATION_POINT_ATTR(k) + (F * (d_target[(x1*dim)+k] - d_target[(x2*dim)+k])) + (F * (d_target[(x3*dim)+k] - d_target[(x4*dim)+k]));
#endif


    int j;
    int mutateIndx = curand(state) % dim;

    ///////////////////// Mutation and Crossover ////////////////
#if CROSSOVER == CROSSOVER_EXP
    bool canMutate = true;
    for (int k = 0; k < dim; k++) {
        if (canMutate) {
            MUTATE();
            canMutate = curand(state) % 1000) >= CR; 
        } else {
            d_trial[(idx*dim)+k] = d_target[(idx*dim)+k];
        }
    }
#else
    for (int k = 0; k < dim; k++) {
        if ((curand(state) % 1000) < CR || k == mutateIndx) {
            MUTATE();
        } else {
            d_trial[(idx*dim)+k] = d_target[(idx*dim)+k];
        } // end if else for creating trial vector
    } // end for loop through parameters
#endif


    float score = costFunc(&d_trial[idx*dim], costArgs, dim);
    if (score < d_cost[idx]) {
        // copy trial into new vector
        for (j = 0; j < dim; j++) {
            d_target2[(idx*dim) + j] = d_trial[(idx*dim) + j];
            //printf("idx = %d, d_target2[%d] = %f, score = %f\n", idx, (idx*dim)+j, d_trial[(idx*dim) + j], score);
        }
        d_cost[idx] = score;
    } else {
        // copy target to the second vector
        for (j = 0; j < dim; j++) {
            d_target2[(idx*dim) + j] = d_target[(idx*dim) + j];
            //printf("idx = %d, d_target2[%d] = %f, score = %f\n", idx, (idx*dim)+j, d_trial[(idx*dim) + j], score);
        }
    }
} // end differentialEvolution function.


// This is the HOST function that handles the entire Differential Evolution process.
// This function handles the entire differentialEvolution, and calls the needed kernel functions.
// @param d_target - a device array with the current agents parameters (requires array with size popSize*dim)
// @param d_trial - a device array with size popSize*dim (worthless outside of function)
// @param d_cost - a device array with the costs of the last generation afterwards size: popSize
// @param d_target2 - a device array with size popSize*dim (worthless outside of function)
// @param d_min - a list of the minimum values for the set of parameters (size = dim)
// @param d_max - a list of the maximum values for the set of parameters (size = dim)
// @param h_cost - this function once the function is completed will contain the costs of final generation.
// @param randStates - an array of random number generator states. Array created using createRandNumGen funtion
// @param dim - the number of dimensions the equation being minimized has.
// @param popSize - this the population size for DE, or otherwise the number of agents that DE will use. (see DE paper for more info)
// @param maxGenerations - the max number of generations DE will perform (see DE paper for more info)
// @param CR - Crossover Constant used by DE (see DE paper for more info)
// @param F - the scaling factor used by DE (see DE paper for more info)
// @param costArgs - this a set of any arguments needed to be passed to the cost function. (must be in device memory already)
// @param h_output - the host output vector of function
float differentialEvolution(float *d_target,
                           float *d_trial,
                           float *d_cost,
                           float *d_target2,
                           float *d_min,
                           float *d_max,
                           float *h_cost,
                           void *randStates,
                           int dim,
                           int popSize,
                           int maxGenerations,
                           int CR, // Must be given as value between [0,999]
                           float F,
                           void *costArgs,
                           float *h_output)
{
    cudaError_t ret;
    int power32 = ceil(popSize / 32.0) * 32;

    // Allocate mutation indices
    int *currentMutationIndices, *nextMutationIndices;
    cudaMalloc(&currentMutationIndices, sizeof(int) * popSize * MUTATION_INDICES_COUNT);
    cudaMalloc(&nextMutationIndices, sizeof(int) * popSize * MUTATION_INDICES_COUNT);

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    float *bestElement;
    cudaMalloc(&bestElement, sizeof(float) * dim);

    // generate random vector
    generateRandomVectorAndInit<<<1, power32, 0, streams[0]>>>(d_target, d_min, d_max, d_cost,
                    costArgs, (curandState_t *)randStates, popSize, dim, clock());
    generateMutationIndices<<<1, power32, 0, streams[1]>>>(popSize, (curandState_t *)randStates, currentMutationIndices);
    gpuErrorCheck(cudaPeekAtLastError());

#if MUTATION_POINT == MUTATION_POINT_BEST
    findBest<<<1, power32, (sizeof(float) + sizeof(int)) * popSize * 2, streams[0]>>>(d_target, d_cost, popSize, dim, bestElement);
    gpuErrorCheck(cudaPeekAtLastError());
#endif

    for (int i = 1; i <= maxGenerations; i++) {
    #if MUTATION_POINT == MUTATION_POINT_BEST
        findBest<<<1, power32, (sizeof(float) + sizeof(int)) * popSize * 2, streams[0]>>>(d_target, d_cost, popSize, dim, bestElement);
    #endif
        generateMutationIndices<<<1, power32, 0, streams[1]>>>(popSize, (curandState_t *)randStates, nextMutationIndices);

        // start kernel for this generation
       evolutionKernel<<<1, power32, 0, streams[0]>>>(d_target, bestElement, d_trial, d_cost, d_target2, currentMutationIndices, d_min, d_max,
              (curandState_t *)randStates, dim, popSize, CR, F, costArgs);

        gpuErrorCheck(cudaPeekAtLastError());

        // swap buffers, places newest data into d_target.
        float *tmp_target = d_target;
        d_target = d_target2;
        d_target2 = tmp_target;

        int* tmp_indices = nextMutationIndices;
        nextMutationIndices = currentMutationIndices;
        currentMutationIndices = tmp_indices;
    } // end for (generations)

    ret = cudaDeviceSynchronize();
    gpuErrorCheck(ret);

    ret = cudaMemcpy(h_cost, d_cost, popSize * sizeof(float), cudaMemcpyDeviceToHost);
    gpuErrorCheck(ret);

    // find min of last evolutions
    float bestCost = FLT_MAX;
    for (int i = 0; i < popSize; i++) {
        float curCost = h_cost[i];
        if (curCost <= bestCost) {
            bestCost = curCost;
        }
    }

    return bestCost;
}

// allocate the memory needed for random number generators.
void *createRandNumGen(int size)
{
    void *x;
    gpuErrorCheck(cudaMalloc(&x, sizeof(curandState_t)*size));
    return x;
}
