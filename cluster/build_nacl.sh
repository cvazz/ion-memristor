
exit_on_error() {
    exit_code=$1
    last_command=${@:2}
    if [ $exit_code -ne 0 ]; then
        >&2 echo "\"${last_command}\" command failed with exit code ${exit_code}."
        exit $exit_code
    fi
}

###############################################################################
################################# Variable ####################################
###############################################################################

DEBUG="true"
DEBUG="false"
sidelength=10
N_ion=${1:-0}
N_ion=$(( $N_ion  * $sidelength * $sidelength / 10 / 10 ))
charge=${2:-1}
water_density=${3:-23.9} #nm
force_new_folder=${4:-1}
emax=(0 0.5 1 2 3 4 5)
emax=(0 )

###############################################################################
############################### Sanity Checks #################################
###############################################################################
if [ $N_ion == 0 ] ; 
then 
    echo "currently N_ion variable"
    exit
fi
if [ $charge -lt 0 ] || [ $charge -gt 2 ] ; then
    echo "Why charge weird?"
    exit
fi


###############################################################################
################################# Files #######################################
###############################################################################
creator="create.gen2.lmp"
execfile=" exec.generic.lmp"
bashfile="run_batch.sh"
bashcrea="run_init.sh"
###############################################################################
################################## Status #####################################
###############################################################################

statusstring="N_ion: $N_ion, Charge: $charge e,"
statusstring2=" Water Density $water_density nm "
repeat=$(seq 1 ${#statusstring})
under=$(printf '_%.0s' $repeat)
empty=$(printf ' %.0s' $repeat)
echo " $under"; echo "|$empty|"
echo "|$statusstring|"; 
echo "|$statusstring2|"; 
echo "|$under|"; echo ""

###############################################################################
############################## Specificiation #################################
###############################################################################

iterate=() 

for ii in ${!emax[@]}; do 
   iterate[$ii]="-var E_max ${emax[$ii]} "
done

echo "Number of arguments ${#iterate[@]}"

###############################################################################
################################## Naming #####################################
###############################################################################

wd=$(printf %.0f "$(echo $water_density*100 | bc -l)")
# (echo $*100 | bc -l)
specific="nacl-n$N_ion-c$charge-s$sidelength-rho$wd"
folname="md-$specific"
jobname="$specific"

sfile="rs.nc_$specific"
sf_folder="/nethome/7036795/experiments/start_files"
startfile=$sf_folder/$sfile


###############################################################################
############################## Folder Prep ####################################
###############################################################################

if [ $DEBUG != "true" ]; then
    orgfol="../data/$( date '+%F' )"
    tarfol="$orgfol/$folname"


    mkdir -p $orgfol
    mkdir $tarfol
    new_folder=$?
    if  [ $new_folder -ne 0 ] && [ "$force_new_folder" -eq 1 ] 
    then
        echo "Folder exists already"
        exit $new_folder
    fi
fi


###############################################################################
########################## Make/Get Start Config  #############################
###############################################################################

lmp_command="/nethome/7036795/lammps2019/src/lmp_mpi"
suploc="/nethome/7036795/experiments/supple"
var_loc="variable        sup_folder index $suploc"
echo $tarfol
cp $execfile $bashfile $bashcrea $tarfol
exit_on_error $? !!
sge_commands=""
if [ ! -f $startfile ] ; then
    cp $creator $tarfol
    cd $tarfol
    echo "creating new start file"
    creargs=""
    creargs="$creargs -var sidelength $sidelength"
    creargs="$creargs -var water_density $water_density"
    creargs="$creargs -var N_ion $N_ion"
    creargs="$creargs -var charge $charge"
    temp=$creator;
    echo $var_loc | cat - $temp > temp && mv temp $temp
    job_name="cr-$specific"
    crecommand="qsub -N $job_name $bashcrea $creator '$creargs' $sfile $sf_folder"
    sge_commands="-hold_jid $job_name"
    echo $crecommand      
    if [ $DEBUG != "true" ]; then eval $crecommand; fi
else
    echo "using existing start file"
    cp $startfile $tarfol
    cd $tarfol
fi

###############################################################################
################################ Execution ####################################
###############################################################################

temp=$execfile;
echo $var_loc | cat - $temp > temp && mv temp $temp
sge_commands="$sge_commands -t 1-${#iterate[@]}"
sge_commands="$sge_commands -N $specific"
command_without_array="qsub $sge_commands $bashfile $execfile $sfile '$bash_args'" 
echo "$command_without_array"
echo "${iterate[@]}"
command='qsub $sge_commands $bashfile $execfile $sfile "$bash_args" "${iterate[@]}"'
if [ $DEBUG != "true" ]; then eval $command; fi
#eval $command      

echo "Simulation in Folder: $tarfol"
if [ $DEBUG == "true" ]; then echo "!!!!!!!!!!"; echo "DEBUG SESSION"; echo "!!!!!!!!!!"; fi
