initial:
	nvcc -o programInitial main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU_initial.cu

article:
	nvcc -o programArticle main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU_article.cu

clean:
	rm programDE
