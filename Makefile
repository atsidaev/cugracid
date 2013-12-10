CC=mpicc
CXX=mpic++
NVCC=nvcc
NVCFLAGS=-arch=sm_21

LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64 -lcudart

all:	v3

clean:
	find -name '*.o' -delete
	rm v3

v3:	v3.o cuda/info.o cuda/Vz.o grid/Grid.o

%.o:	%.cu
	$(NVCC) -c $(NVCFLAGS) $^ -o $@
