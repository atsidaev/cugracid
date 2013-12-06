CC=mpicc
NVCC=nvcc
NVCFLAGS=-arch=sm_21

LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64 -lcudart

all:	v3

clean:
	rm *.o

v3:	v3.o

%.o:	%.cu
	$(NVCC) -c $(NVCFLAGS) $^ -o $@
