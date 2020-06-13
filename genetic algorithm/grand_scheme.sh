#!/bin/bash
# The optimal workflow is to run 2*40=80 separate array jobs.
# To test the algorithm we try different settings.
BATCH_SIZE=1
PARTS=1
SIMULATIONS_PART=160
# Data of one generation
POPULATION=40
SCENARIOS=4
# It should hold BATCH_SIZE*SIMULATIONS_PART*PARTS = POPULATION*SCENARIOS
START_GENERATION=33
END_GENERATION=33
GUIDES=2
# Bash-files for genetic algorithm
SEEDS="generate_scenarios.sh"
INITIALIZATION="initialize.sh"
EVALUATION="genetic_algorithm.sh"
COLLECTION="gather_results.sh"
PLOT="plot.sh"
# Check that the code below is correct, when you change PARTS
I=${START_GENERATION}
while [ ${I} -le ${END_GENERATION} ]; do
  if [ ${I} -eq ${START_GENERATION} ]
  then
    if [ ${I} -eq "0" ]
    then
      JOBIDX=$(sbatch ${SEEDS} ${POPULATION} ${SCENARIOS})
      JOBID0=$(sbatch --dependency=afterany:${JOBIDX:20:8} ${INITIALIZATION} ${POPULATION} ${SCENARIOS} ${GUIDES})
      JOBIDA=$(sbatch --dependency=afterany:${JOBID0:20:8} --array=0-$((${SIMULATIONS_PART}-1)) ${EVALUATION} ${I} 0 ${GUIDES} ${SIMULATIONS_PART} ${BATCH_SIZE})
      JOBID=$(sbatch --dependency=afterany:${JOBIDA:20:8} ${COLLECTION} ${I} ${POPULATION} ${SCENARIOS} ${GUIDES} ${BATCH_SIZE} ${PARTS} ${SIMULATIONS_PART} ${JOBIDA:20:8})
    else
      JOBIDA=$(sbatch --array=0-$((${SIMULATIONS_PART}-1)) ${EVALUATION} ${I} 0 ${GUIDES} ${SIMULATIONS_PART} ${BATCH_SIZE})
      JOBID=$(sbatch --dependency=afterany:${JOBIDA:20:8} ${COLLECTION} ${I} ${POPULATION} ${SCENARIOS} ${GUIDES} ${BATCH_SIZE} ${PARTS} ${SIMULATIONS_PART} ${JOBIDA:20:8})
    fi
  else
    JOBID2A=$(sbatch --dependency=afterany:${JOBID:20:8} --array=0-$((${SIMULATIONS_PART}-1)) ${EVALUATION} ${I} 0 ${GUIDES} ${SIMULATIONS_PART} ${BATCH_SIZE})
    JOBID=$(sbatch --dependency=afterany:${JOBID2A:20:8} ${COLLECTION} ${I} ${POPULATION} ${SCENARIOS} ${GUIDES} ${BATCH_SIZE} ${PARTS} ${SIMULATIONS_PART} ${JOBID2A:20:8})
  fi
  let I=${I}+1
done
