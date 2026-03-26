CXX=nvcc
CXXFLAGS=-Xcompiler "-O3 -fopenmp"

all: a2 a2_kepler

a2: main.cu
	$(CXX) -O3 $(CXXFLAGS) -o a2 main.cu

clean:
	rm -f a2 a2_kepler a2_eval_kepler knn.txt approx_knn.txt kmeans.txt eval a2_eval_cu
