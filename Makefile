CXX=g++
CXXFLAGS=--std=c++11 -DGEO_BUILD_ALL -g

HIPCC=/opt/rocm/bin/hipcc
HIPCCFLAGS=

NVCC=nvcc

NVCFLAGS+=-gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_37,code=\"sm_37,compute_37\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" -gencode=arch=compute_52,code=\"sm_52,compute_52\"

CC=$(NVCC)
CFLAGS=$(NVCFLAGS)

# Uncomment to use HIP (or comment to not to)
# CC=$(HIPCC)
# CFLAGS=$(HIPCCFLAGS)

#LDFLAGS=-lcudart

BINARIES:=main
APPS:=vz lc m_to_km compare

all: $(BINARIES)

clean:
	find -name '*.o' -delete
	rm $(BINARIES) || true

main:	main.o $(APPS:%=apps/%.o) calc/direct.o calc/golden.o cuda/info.o cuda/Vz.o grid/Grid.o

test: main
	./main vz --test

%.o:	%.cu
	$(CC) -c $(CFLAGS) $^ -o $@

m_to_km:	grid/Grid.o
