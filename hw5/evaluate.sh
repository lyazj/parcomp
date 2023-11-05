#!/bin/bash

for NPROC in $(seq 1 64); do
    echo ${NPROC}
    mpirun ./redist ${NPROC} ./prog 2>/dev/null | tee run_${NPROC}.log
done &
disown
