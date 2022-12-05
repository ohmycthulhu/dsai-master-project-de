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

//
//  testMain.cpp
//
// This is a test code to show an example usage of Differential Evolution

#include <stdio.h>

#include "DifferentialEvolution.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

std::vector<float> executeDE(const size_t population_size, const size_t dim, float* cost) {
    // create the min and max bounds for the search space.
    float minBounds[2] = {-50, -50};
    float maxBounds[2] = {100, 200};

    // data that is created in host, then copied to a device version for use with the cost function.
    struct data x;

    // a random array or data that gets passed to the cost function.    
    x.arr = new float[3]{2.5, 2.6, 2.7};
    x.v = 3;
    x.dim = dim;
    // Create the minimizer with a popsize of 192, 50 generations, Dimensions = 2, CR = 0.9, F = 2

    DifferentialEvolution minimizer(population_size,50, 2, 0.9, 0.5, minBounds, maxBounds);

    return minimizer.fmin(&x, cost);
}

template<typename T>
void print_vector(std::vector<T> vec, const size_t dim, const char* sep=", ") {
    for (int i = 0; i < dim; i++) {
        std::cout << vec[i];
        if (i < dim - 1) {
            std::cout << sep;
        }
    }
    std::cout << std::endl;
}

void print_configuration(size_t sample_size, size_t dim, size_t population_size) {
    std::cout << "Configuration:" << std::endl;
    std::cout << "Population size: " << population_size << std::endl;
    std::cout << "Dimensions: " << dim << std::endl;
    std::cout << "Run count: " << sample_size << std::endl;
    std::cout << std::endl;

    std::cout << "Function: "
    #if COST_SELECTOR == COST_RASTRIGIN
               << "Rastrigin Function"
    #elif COST_SELECTOR == COST_RASTRIGIN
               << "Rosenblock Function"
    #elif COST_SELECTOR == COST_GRIEWANK
               << "Griewank Function"
    #elif COST_SELECTOR == COST_GRIEWANK
               << "Sphere Function"
    #endif
               << std::endl;

    std::cout << std::endl;

    // Generates: "Strategy: best/2/exp"
    std::cout << "Strategy: ";
    std::cout << (MUTATION_POINT == MUTATION_POINT_BEST ? "best" : "rand") << "/";
    std::cout << MUTATION_PARAMS << "/";
    std::cout << (CROSSOVER == CROSSOVER_EXP ? "exp" : "bin") << std::endl;
    std::cout << std::endl << std::endl;
}

int main(int argc, char** argv)
{
    size_t sample_size = 1;
    if (argc > 1) {
        sample_size = atoi(argv[1]);
    }

    size_t dim = 2;
    if (argc > 2) {
        dim = atoi(argv[2]);
    }

    size_t population_size = 192;
    if (argc > 3) {
        population_size = atoi(argv[3]);
    }

    print_configuration(sample_size, dim, population_size);

    std::vector<float> result;
    float cost;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    if (sample_size > 1) {
        std::cout << "Execution results:" << std::endl;

        std::cout << "t (ms)" << "\t" << "C" << std::endl;

        for (int i = 0; i < sample_size; i++) {
            start = std::chrono::high_resolution_clock::now();
            result = executeDE(population_size, generations_count, dim, &cost);
            end = std::chrono::high_resolution_clock::now();
            auto millis = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
 
            std::cout << millis << "\t" << cost << std::endl;
        }
    } else {
        result = executeDE(population_size, dim, &cost);
        print_vector(result, dim);

        std::cout << "Best cost: " << cost << std::endl;
    }

    // get the result from the minimizer
    std::cout << "Finished main function." << std::endl;
    return 0;
}