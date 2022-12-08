#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -q all.q
#$ -pe mpich 4

# ./mc.out < run.inp


run_file="${1:-exec.efield.lmp}"
args="$2"
echo $run_file
echo $args

command_cluster="/nethome/7036795/lammps2019/src/lmp_mpi"


multi_core="mpiexec -np ${NSLOTS}"
if [ "$multi_core" = "mpiexec -np " ]; then
    echo "no multi core setup"
    multi_core=""
fi

START=$(date +%s.%N)

run_command="$multi_core $command_cluster < $run_file $args" 
echo "$run_command"
eval $run_command 

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "Time elapsed: $DIFF"

