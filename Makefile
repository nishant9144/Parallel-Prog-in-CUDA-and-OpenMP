CXX=nvcc
CXXFLAGS=-Xcompiler "-fopenmp"

all: a2

a2: main.cu
	$(CXX) $(CXXFLAGS) -o a2 main.cu

clean:
	rm -f a2 knn.txt approx_knn.txt kmeans.txt
