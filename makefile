initial:
	nvcc -o programInitial main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU_initial.cu

article:
	nvcc -o programArticle main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU_article.cu

sequential:
	nvcc -o sequential main_sequential.cpp DifferentialEvolution_sequential.cpp DifferentialEvolutionGPU_sequential.cu

clean:
	rm programDE
