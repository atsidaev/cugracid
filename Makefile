CXX=g++
CXXFLAGS=--std=c++11 -DGEO_BUILD_ALL -g

HIPCC=/opt/rocm/bin/hipcc
HIPCCFLAGS=

NVCC=nvcc

#CC=$(HIPCC)
CC=$(NVCC)
NVCFLAGS+=-gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_37,code=\"sm_37,compute_37\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" -gencode=arch=compute_52,code=\"sm_52,compute_52\"

LDFLAGS=-lcudart

BINARIES:=main

all:	$(BINARIES)

clean:
	find -name '*.o' -delete
	rm $(BINARIES)

main:	main.o v3.o lc.o m_to_km.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

v3:	v3.o direct.o cuda/info.o cuda/Vz.o grid/Grid.o

lc:	lc.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

recalc: recalc.o recalc_up.o cuda/info.o grid/Grid.o cuda/recalc.o

%.o:	%.cu
#	$(HIPCC) -c $(HIPCCFLAGS) $^ -o $@
	$(NVCC) -c $(NVCFLAGS) $^ -o $@

m_to_km:	grid/Grid.o
