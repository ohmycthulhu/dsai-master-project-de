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
// float fooCost(const float *vec, const void *args)
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

// for random numbers in a kernel
#include "DifferentialEvolutionGPU.h"

#include <curand_kernel.h>


#include <cuda_runtime.h>

// for FLT_MAX
#include <cfloat>

#include <iostream>

// for clock()
#include <ctime>
#include <cmath>
#include <random>

// basic function for exiting code on CUDA errors.
// Does no special error handling, just exits the program if it finds any errors and gives an error message.
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

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
float rastriginFunc(const float *vec, const void *args, const int dim)
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
float rosenblockFunc(const float *vec, const void *args, const int dim)
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
float griewankFunc(const float *vec, const void *args, const int dim)
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
float sphereFunc(const float *vec, const void *args, const int dim)
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
float costFunc(const float *vec, const void *args, const int dim) {
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

void generateRandomVectorAndInit(float *d_x, float *d_min, float *d_max,
            float *d_cost, void *costArgs,
            int popSize, int dim, unsigned long seed)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int idx = 0; idx < popSize; idx++) {
        for (int i = 0; i < dim; i++) {
            d_x[(idx*dim) + i] = (dis(gen) * (d_max[i] - d_min[i])) + d_min[i];
        }

        d_cost[idx] = costFunc(&d_x[idx*dim], costArgs, dim);
    }

}

/*
 * Generates 3 non-equal indices for usage in the mutation
 * @param popSize - the population size
 * @param randStates - an array of random number generator states. Array created using createRandNumGen function
 * @param output - a device array used for output
 */
void generateMutationIndices(
        int popSize,
        int* output
) {
    int a, b, c;
    #if MUTATION_PARAMS != MUTATION_PARAMS_1
        int d, f;
    #endif

    for (int idx = 0; idx < popSize; idx++) {
        //////////////////// Random index mutation generation //////////////////
        // select a different random number then index
        do { a = rand() % popSize; } while (a == idx);
        do { b = rand() % popSize; } while (b == idx || b == a);
        do { c = rand() % popSize; } while (c == idx || c == a || c == b);

        #if MUTATION_PARAMS != MUTATION_PARAMS_1
            do { d = rand() % popSize; } while (d == idx || d == a || d == b || d == c);
            do { f = rand() % popSize; } while (f == idx || f == a || f == b || f == c || f == d);
        #endif

        output[idx * MUTATION_INDICES_COUNT] = a;
        output[idx * MUTATION_INDICES_COUNT + 1] = b;
        output[idx * MUTATION_INDICES_COUNT + 2] = c;

        #if MUTATION_PARAMS != MUTATION_PARAMS_1
            output[idx * MUTATION_INDICES_COUNT + 3] = d;
            output[idx * MUTATION_INDICES_COUNT + 4] = f;
        #endif
    }
}

void findBest(float* population, float* costs, int popSize, int dim, float* output) {
    int bestIndex = 0, bestCost = costs[0];

    for (int i = 0; i < popSize; i++) {
        if (bestCost > costs[i]) {
            bestCost = costs[i];
            bestIndex = i;
        }
    }

    for (int i = 0; i < dim; i++) {
        output[i] = population[(bestIndex * dim) + i];
    }
}

void mutationAndCrossover(float* population, float* best, int* mutationIndices, int idx, int dim, int CR, float F, float* output) {
    #if MUTATION_POINT == MUTATION_POINT_BEST
        #define MUTATION_POINT_ATTR(i) best[i]
    #else    
        int a = mutationIndices[idx * MUTATION_INDICES_COUNT];
        #define MUTATION_POINT_ATTR(i) population[(a*dim) + i]
    #endif

    #if MUTATION_PARAMS == MUTATION_PARAMS_1
        int x1 = mutationIndices[idx * MUTATION_INDICES_COUNT + 1];
        int x2 = mutationIndices[idx * MUTATION_INDICES_COUNT + 2];

        #define MUTATE(idx, k) (MUTATION_POINT_ATTR(k) + (F * (population[(x1*dim)+k] - population[(x2*dim)+k])));
    #else
        int x1 = mutationIndices[idx * MUTATION_INDICES_COUNT + 1];
        int x2 = mutationIndices[idx * MUTATION_INDICES_COUNT + 2];
        int x3 = mutationIndices[idx * MUTATION_INDICES_COUNT + 3];
        int x4 = mutationIndices[idx * MUTATION_INDICES_COUNT + 4];

        #define MUTATE() (MUTATION_POINT_ATTR(k) + (F * (population[(x1*dim)+k] - population[(x2*dim)+k])) + (F * (population[(x3*dim)+k] - population[(x4*dim)+k])));
    #endif

    int mutateIndx = rand() % dim;

    ///////////////////// Mutation and Crossover ////////////////
    #if CROSSOVER == CROSSOVER_EXP
        bool canMutate = true;
        for (int k = 0; k < dim; k++) {
            if (canMutate) {
                output[idx * dim + k] = MUTATE();
                canMutate = rand() % 1000) >= CR; 
            } else {
                output[idx * dim + k] = population[(idx*dim)+k];
            }
        }
    #else
        for (int k = 0; k < dim; k++) {
            if ((rand() % 1000) < CR || k == mutateIndx) {
                output[idx * dim + k] = MUTATE();
            } else {
                output[idx * dim + k] = population[(idx*dim)+k];
            } // end if else for creating trial vector
        } // end for loop through parameters
    #endif
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
void evolutionKernel(float *d_target,
                                float *d_best,
                                float *d_trial,
                                float *d_cost,
                                float *d_target2,
                                int *mutationIndices,
                                float *d_min,
                                float *d_max,
                                int dim,
                                int popSize,
                                int CR, // Must be given as value between [0,999]
                                float F,
                                void *costArgs)
{
    int idx;
    int j;
    for (idx = 0; idx < popSize; idx++) {
        mutationAndCrossover(d_target, d_best, mutationIndices, idx, dim, CR, F, d_trial);

        float score = costFunc(&d_trial[idx*dim], costArgs, dim);
        if (score < d_cost[idx]) {
            // copy trial into new vector
            for (j = 0; j < dim; j++) {
                d_target2[(idx*dim) + j] = d_trial[(idx*dim) + j];
            }
            d_cost[idx] = score;
        } else {
            // copy target to the second vector
            for (j = 0; j < dim; j++) {
                d_target2[(idx*dim) + j] = d_target[(idx*dim) + j];
            }
        }
    }
} // end differentialEvolution function.


// This is the HOST function that handles the entire Differential Evolution process.
// This function handles the entire differentialEvolution, and calls the needed kernel functions.
// @param d_target - an array with the current agents parameters (requires array with size popSize*dim)
// @param d_trial - an array with size popSize*dim (worthless outside of function)
// @param d_cost - an array with the costs of the last generation afterwards size: popSize
// @param d_target2 - an array with size popSize*dim (worthless outside of function)
// @param d_min - a list of the minimum values for the set of parameters (size = dim)
// @param d_max - a list of the maximum values for the set of parameters (size = dim)
// @param h_cost - this function once the function is completed will contain the costs of final generation.
// @param randStates - an array of random number generator states. Not used.
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
    srand(time(NULL));
    // Allocate mutation indices
    int *currentMutationIndices = new int[popSize * MUTATION_INDICES_COUNT];

    float *bestElement = new float[dim];

    // generate random vector
    generateRandomVectorAndInit(d_target, d_min, d_max, d_cost, costArgs, popSize, dim, clock());

#if MUTATION_POINT == MUTATION_POINT_BEST
    findBest(d_target, d_cost, popSize, dim, bestElement);
#endif

    for (int i = 1; i <= maxGenerations; i++) {
    #if MUTATION_POINT == MUTATION_POINT_BEST
        findBest(d_target, d_cost, popSize, dim, bestElement);
    #endif
        generateMutationIndices(popSize, currentMutationIndices);

        // start kernel for this generation
        evolutionKernel(d_target, bestElement, d_trial, d_cost, d_target2, currentMutationIndices, d_min, d_max, dim, popSize, CR, F, costArgs);

        // swap buffers, places newest data into d_target.
        float *tmp_target = d_target;
        d_target = d_target2;
        d_target2 = tmp_target;
    } // end for (generations)

    // find min of last evolutions
    float bestCost = FLT_MAX;
    for (int i = 0; i < popSize; i++) {
        float curCost = d_cost[i];
        if (curCost <= bestCost) {
            bestCost = curCost;
        }
    }

    return bestCost;
}


