CC=mpicc
CXX=mpic++
NVCC=nvcc
NVCFLAGS=-arch=sm_21

LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64 -lcudart

all:	v3 lc  m_to_km

clean:
	find -name '*.o' -delete
	rm v3

v3:	v3.o direct.o cuda/info.o cuda/Vz.o grid/Grid.o

lc:	lc.o direct.o cuda/info.o cuda/Vz.o grid/Grid.o

%.o:	%.cu
	$(NVCC) -c $(NVCFLAGS) $^ -o $@

m_to_km:	grid/Grid.o