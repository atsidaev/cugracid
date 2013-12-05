NVCC=nvcc
NVCFLAGS=-c -arch=sm_21

LDFLAGS=-L/opt/cuda/lib64 -lcudart

all:	v3

clean:
	rm *.o

v3:	v3.o

%.o:	%.cu
	$(NVCC) $(NVCFLAGS) $^ -o $@
