CC=mpicc
CXX=mpic++
CXXFLAGS=-DUSE_MPI
NVCC=nvcc
NVCFLAGS=-arch=sm_21

LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64 -lcudart

all:	v3 lc lc_ass m_to_km recalc

clean:
	find -name '*.o' -delete
	rm v3

v3:	v3.o direct.o cuda/info.o cuda/Vz.o grid/Grid.o

lc:	lc.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

lc_ass:	lc_ass.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

recalc: recalc.o recalc_up.o cuda/info.o grid/Grid.o cuda/recalc.o

%.o:	%.cu
	$(NVCC) -c $(NVCFLAGS) $^ -o $@

m_to_km:	grid/Grid.o