#!/bin/bash

#$ -cwd

#$ -V

#$ -N efield

#$ -t 1-20

# STEP=$SGE_TASK_ID

# rho=`echo "$SGE_TASK_ID * 0.0012 + 0.0288" | bc -l`
# mkdir state_$STEP
# cp mc_sc_ranlist.f90 run.inp lat.inp state_$STEP
# cd state_$STEP
# sed -i "s/denlat/$rho/g" lat.inp



# ./mc.out < run.inp

emax=(0 1 5 10 20 50)
fileloc="./output-efield.csv"
run_file="in.slit.efield.bsh"
header="time,temp,density,ion_count,efield,current_x,current_y"

command_home="lmp_stable -in " 
command_cluster="lmp_stable -in "
run_command="$command_home $run_file"
# csv location
save2file="-var thermo_file $fileloc "
echo $header > $fileloc

for ee in ${emax[@]}
do 
    emax_var="-var E_max $ee"
    # echo "lmp_stable -in $run $efield_str $save2file"
    # lmp_stable -in $run $efield_str $save2file
    echo $run_command $emax_var $save2file
done

# qsub -N test -cwd -j y -V -pe mpich_mod 4 -q all.q run.txt
echo ${#emax[@]}
echo ${emax[2]}
echo ${emax[5]}