CXX=nvcc
CXXFLAGS=-Xcompiler "-O3 -fopenmp"

all: a2

a2: main.cu
	$(CXX) -O3 $(CXXFLAGS) -o a2 main.cu

clean:
	rm -f a2 knn.txt approx_knn.txt kmeans.txt eval a2_eval_cu
