# Bitonic Sort Implementations
**Name:** D B Y Binuwara - IT23184558  
**GitHub:** https://github.com/D-B-Yesen-Binuwara/Bitonic-Sort-IT23184558_OpenMP.git

This repository contains multiple implementations of the Bitonic Sort algorithm:
- **Serial**: Basic sequential implementation
- **OpenMP**: Parallel implementation using OpenMP
- **MPI**: Distributed implementation using MPI
- **CUDA**: GPU implementation using CUDA

## Prerequisites (WSL/Linux)
```bash
# Install OpenMP
sudo apt update
sudo apt install libomp-dev

# Install MPI
sudo apt install mpich
```

## Serial Version
```bash
cd Serial
gcc -O2 bitonic.c -o bitonic

# Windows:
bitonic.exe [array_size]

# WSL/Linux:
./bitonic [array_size]
```

## OpenMP Version
```bash
cd OpenMP
gcc -fopenmp -O2 bitonicOmp02.c -o bitonicOmp02

# Windows:
bitonicOmp02.exe [array_size] [num_threads]

# WSL/Linux:
./bitonicOmp02 [array_size] [num_threads]

# Test runs:
./bitonicOmp02 100000 1
./bitonicOmp02 100000 2
./bitonicOmp02 100000 4
./bitonicOmp02 100000 8
```

## MPI Version
```bash
cd MPI
mpicc -O2 bitonicMPI_fixed.c -o bitonicMPI_fixed

# Windows:
mpiexec -n [num_processes] bitonicMPI_fixed.exe [array_size]

# WSL/Linux:
mpirun -np [num_processes] ./bitonicMPI_fixed [array_size]
mpirun --oversubscribe -np 32 ./bitonicMPI_fixed 100000
```

## CUDA Version

### Windows (with CUDA Toolkit)
```bash
cd CUDA
nvcc -O2 bitonicCUDA.cu -o bitonicCUDA
bitonicCUDA.exe [array_size] [threads_per_block]
```

### Google Colab
```bash
%%bash
nvcc -O3 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  bitonicCUDA02.cu -o bitonicCUDA02

%%bash
./bitonicCUDA02 8192 256
```
