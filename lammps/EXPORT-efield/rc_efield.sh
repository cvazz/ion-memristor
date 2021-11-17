#!/bin/bash

#$ -cwd

#$ -V

#$ -N efield

#$ -t 0-2

# ./mc.out < run.inp

emax=(0 1 5)
STEP=$SGE_TASK_ID
# STEP=2


fileloc="./efield.csv"
run_file="in.slit.efield.bsh"
header="time,temp,density,ion_count,efield,current_x,current_y"

command_home="lmp_stable -in " 
command_cluster="lmp_stable -in "
run_command="$command_home $run_file"
# csv location
save2file="-var thermo_file $fileloc "
echo $header > $fileloc


ee=${emax[$STEP]}
emax_var="-var E_max $ee"
echo $run_command $emax_var $save2file


# mpiexec -np ${NSLOTS} /nethome/campo007/LAMMPS_3Mar/lammps-stable_3Mar2020/build/lmp_marversion < lammps_rods.inp
# # qsub -N test -cwd -j y -V -pe mpich_mod 4 -q all.q run.txt
# WhySoSerious??
# WhySoSerious??
# WhySoSerious??
# #!/bin/bash
# mpiexec -np ${NSLOTS}  /nethome/7036795/lammps2019/src/lmp_mpi < run_lj.inp

# git clone -b stable_5Jun2019 https://github.com/lammps/lammps.git lammps2


