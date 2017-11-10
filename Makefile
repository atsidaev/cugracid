CC=nvcc
CXX=g++
CXXFLAGS=-I$(CUDA_INSTALL_PATH)/include --std=c++11 -DGEO_BUILD_ALL
NVCC=nvcc
NVCFLAGS=-gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -x cu $(shell pkg-config --cflags cuda-9.0)

LDFLAGS=$(shell pkg-config --libs cuda-9.0) -lcudart

all:	v3 lc lc_ass m_to_km recalc

clean:
	find -name '*.o' -delete
	rm v3 lc lc_ass recalc m_to_km

v3:	v3.o direct.o cuda/info.o cuda/Vz.o grid/Grid.o

lc:	lc.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

lc_ass:	lc_ass.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

recalc: recalc.o recalc_up.o cuda/info.o grid/Grid.o cuda/recalc.o

%.o:	%.cu
	$(NVCC) -c $(NVCFLAGS) $^ -o $@

m_to_km:	grid/Grid.o
