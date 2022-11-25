initial:
	nvcc -o programInitial main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU_initial.cu

clean:
	rm programDE
