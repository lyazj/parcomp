#!/bin/bash

for NPROC in 4 16 64; do
	for LOG_NPART in 2 4 8; do
		for LOG_NSTEP in 2 4 8; do
			echo ${NPROC} ${LOG_NPART} ${LOG_NSTEP}
			./Evaluate 1 ${NPROC} ${LOG_NPART} ${LOG_NSTEP} 2>/dev/null \
				| tee run_${NPROC}_${LOG_NPART}_${LOG_NSTEP}.log
		done
	done
done
