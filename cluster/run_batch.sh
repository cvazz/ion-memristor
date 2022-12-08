#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -q all.q
#$ -pe mpich 4
# ./mc.out < run.inp


fileloc="./output-efield.csv"
run_file="${1:-exec.efield.lmp}"
load_file="${2:-rs.ew100}"
echo "running: $run_file"
echo "loading from file: $load_file"

let STEP=$SGE_TASK_ID-1
echo STEP: $STEP
if [ $STEP -lt 0 ] ;then
    STEP=1
fi

args=$3
shift 3

array=("$@")

itarg="${array[$STEP]}"

command_cluster="/nethome/7036795/lammps2019/src/lmp_mpi"

# csv location
loadFromFile="-var in_name $load_file"

if [ "$loadFromFile" = "-var in_name " ]; then 
    loadFromFile=""
fi

echo "constant args: $args"
args="$args $save2file $loadFromFile  "
args="$args $itarg"
echo "all args: $args"

multi_core="mpiexec -np ${NSLOTS}"
if [ "$multi_core" = "mpiexec -np " ]; then
    echo "no cores"
    multi_core=""
elif [ ${NSLOTS} -le 1 ]; then
    echo "One core or less: no multi core setup"
    multi_core=""
fi
echo "multi: $multi_core"



START=$(date +%s.%N)
run_command="$multi_core $command_cluster < $run_file $args" 

echo "$run_command"
eval $run_command 

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "Time elapsed: $DIFF"

