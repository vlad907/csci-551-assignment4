# Build on ECC (or any Linux box) with MPI: module load mpi/openmpi or use mpicc from Intel MPI.
# Example: make && mpirun -np 4 ./train_ex4 --dt 0.0001

MPICC ?= mpicc
CFLAGS = -O3 -std=c99 -Wall -Wextra -I../functiongen
LDFLAGS = -lm

train_ex4: train_ex4.c ../functiongen/ex4.h
	$(MPICC) $(CFLAGS) -o train_ex4 train_ex4.c $(LDFLAGS)

.PHONY: clean
clean:
	rm -f train_ex4
