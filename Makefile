CC      = gcc
NVCC    = nvcc

#CFLAGS  = -Wall -Wextra -O2 -g -std=c11
CFLAGS = -Wall -Wextra -O2 -g -std=c11 -I$(CUDA_HOME)/include
NVFLAGS = -O2 -g

SRCS_C  = bwa_build.c fm_index.c bwa_smem.c ksw.c
SRCS_CU = bwa_sw.cu

OBJS = $(SRCS_C:.c=.o) $(SRCS_CU:.cu=.o)

.PHONY: all test clean

all: libbwa.a

libbwa.a: $(OBJS)
	ar rcs $@ $^

test: main
	./main

main: main.c $(OBJS)
	$(NVCC) -o $@ $^ -lz

%.o: %.c bwa.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu bwa.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) libbwa.a main