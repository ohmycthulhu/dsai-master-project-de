parallel: parallel.cpp DifferentialEvolution.hpp DifferentialEvolutionGPU.h parallel/*
	nvcc -o de_parallel parallel.cpp parallel/DifferentialEvolution.cpp parallel/DifferentialEvolutionGPU.cu

sequential: sequential.cpp DifferentialEvolution.hpp DifferentialEvolutionGPU.h sequential/*
	nvcc -o de_sequential sequential.cpp sequential/DifferentialEvolution.cpp sequential/DifferentialEvolutionGPU.cu

clean:
	rm de_parallel de_sequential
