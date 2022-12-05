gpu:
	nvcc -o programGPU main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU_article.cu

sequential:
	nvcc -o programSequential main_sequential.cpp DifferentialEvolution_sequential.cpp DifferentialEvolutionGPU_sequential.cu

clean:
	rm programGPU programSequential