#! /bin/bash


#SBATCH --job-name=slurm-imagingem
#SBATCH --output=log.txt
#SBATCH --ntasks=100
#SBATCH --mem=175G
#SBATCH --time=1:00:00
#SBATCH --partition=mem



module purge
module load anaconda3/2020.02/gcc-9.2.0
source activate lowrankrfi


for rfi in -10 -5 -3 0 3 5 10 
do
    srun -n 100 python robustimaging.py --PRFI $rfi --SNR $snr
    wait

    for i in {0..99}
    do
        h5copy -i MC_$i.hdf5 -o PRFI_$rfi.hdf5 -s "MC_$i" -d "PRFI_$rfi/MC_$i" -p
    done
    wait

    rm MC_*

done

for rfi in -10 -5 -3 0 3 5 10   
do
    h5copy -i PRFI_$rfi.hdf5 -o SNR_$snr.hdf5 -s "PRFI_$rfi" -d "SNR_$snr/PRFI_$rfi" -p
done
wait 

rm PRFI_*


