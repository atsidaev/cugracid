CXX=g++
CXXFLAGS=--std=c++11 -DGEO_BUILD_ALL -g

HIPCC=/opt/rocm/bin/hipcc
HIPCCFLAGS=
CC=$(HIPCC)

LDFLAGS=-lcudart

#BINARIES:=v3 lc m_to_km recalc
BINARIES:=main


all:	$(BINARIES)

clean:
	find -name '*.o' -delete
	rm $(BINARIES) recalc

main:	main.o v3.o lc.o m_to_km.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

v3:	v3.o direct.o cuda/info.o cuda/Vz.o grid/Grid.o

lc:	lc.o direct.o golden.o cuda/info.o cuda/Vz.o grid/Grid.o

recalc: recalc.o recalc_up.o cuda/info.o grid/Grid.o cuda/recalc.o

%.o:	%.cu
	$(HIPCC) -c $(HIPCCFLAGS) $^ -o $@
#	$(NVCC) -c $(NVCFLAGS) $^ -o $@

m_to_km:	grid/Grid.o
