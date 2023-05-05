CXX=g++
CXXFLAGS=--std=c++11 -DGEO_BUILD_ALL -g

HIPCC=/opt/rocm/bin/hipcc
HIPCCFLAGS=

NVCC=nvcc

#CC=$(HIPCC)
CC=$(NVCC)
NVCFLAGS+=-gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_37,code=\"sm_37,compute_37\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" -gencode=arch=compute_52,code=\"sm_52,compute_52\"

#LDFLAGS=-lcudart

BINARIES:=main
APPS:=v3 lc m_to_km compare

all: $(BINARIES)

clean:
	find -name '*.o' -delete
	rm $(BINARIES)

main:	main.o $(APPS:%=apps/%.o) calc/direct.o calc/golden.o cuda/info.o cuda/Vz.o grid/Grid.o

%.o:	%.cu
#	$(HIPCC) -c $(HIPCCFLAGS) $^ -o $@
	$(NVCC) -c $(NVCFLAGS) $^ -o $@

m_to_km:	grid/Grid.o
