# Build on cscigpu (Linux) or local CUDA install.
# Single command: make
#
# Override CUDA if needed, e.g.:
#   make CUDA_PATH=/usr/local/cuda-12.9

OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

CUDA_PATH       ?= /usr/local/cuda
ifeq ($(OSUPPER),DARWIN)
  CUDA_PATH     ?= /usr/local/cuda
endif

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

NVCC  ?= $(CUDA_BIN_PATH)/nvcc
GCC   ?= gcc

EXTRA_NVCCFLAGS ?= -Wno-deprecated-gpu-targets
LDFLAGS := -L$(CUDA_LIB_PATH) -lcudart -lm
INCLUDES := -I$(CUDA_INC_PATH) -I.

ifeq ($(dbg),1)
  NVCCFLAGS += -g -G
  CCFLAGS += -g
endif

.PHONY: all clean

all: cuda_trap1 trap_seq

cuda_trap1.o: cuda_trap1.cu timer.h
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) -c -o $@ cuda_trap1.cu

cuda_trap1: cuda_trap1.o
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)

trap_seq.o: trap_seq.c timer.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -c -o $@ trap_seq.c

trap_seq: trap_seq.o
	$(GCC) $(CCFLAGS) -o $@ $+ -lm

clean:
	rm -f cuda_trap1 cuda_trap1.o trap_seq trap_seq.o
