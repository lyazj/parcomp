#!/bin/bash

for NPROC in 1 2 4 6 8 16 32 64; do
    echo ${NPROC}
    mpirun ./fft_mpi ${NPROC} 2>/dev/null | tee run_${NPROC}.log
done
