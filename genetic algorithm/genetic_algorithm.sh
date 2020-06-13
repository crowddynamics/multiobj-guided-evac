#!/bin/bash
#SBATCH -n 1
##CHECK THE TIME
#SBATCH -t 00:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --constraint=ivb

module load anaconda3
source activate multiobj-guided-evac

generation=$1
part=$2
guides=$3
simulations_part=$4
batch_size=$5

SIMULATION_ID=$(($SLURM_ARRAY_TASK_ID*$batch_size+$simulations_part*$part))

scenarios="scenarios.txt"
scenarios=$(cat "$scenarios")
scenarios=($scenarios)

# First gene data
cells1="cells1_${generation}.txt"
randcell1=$(cat "$cells1")
randcell1=($randcell1)

exits1="exits1_${generation}.txt"
randexit1=$(cat "$exits1")
randexit1=($randexit1)

if [ $guides -ge "2" ]
then
  # Second gene data
  cells2="cells2_${generation}.txt"
  randcell2=$(cat "$cells2")
  randcell2=($randcell2)

  exits2="exits2_${generation}.txt"
  randexit2=$(cat "$exits2")
  randexit2=($randexit2)

  if [ $guides -ge "3" ]
  then
    # Third gene data
    cells3="cells3_${generation}.txt"
    randcell3=$(cat "$cells3")
    randcell3=($randcell3)

    exits3="exits3_${generation}.txt"
    randexit3=$(cat "$exits3")
    randexit3=($randexit3)

    if [ $guides -ge "4" ]
    then
      # Fourth gene data
      cells4="cells4_${generation}.txt"
      randcell4=$(cat "$cells4")
      randcell4=($randcell4)

      exits4="exits4_${generation}.txt"
      randexit4=$(cat "$exits4")
      randexit4=($randexit4)

      if [ $guides -ge "5" ]
      then
        # Fifth gene data
        cells5="cells5_${generation}.txt"
        randcell5=$(cat "$cells5")
        randcell5=($randcell5)

        exits5="exits5_${generation}.txt"
        randexit5=$(cat "$exits5")
        randexit5=($randexit5)

        if [ $guides -ge "6" ]
        then
          # Sixth gene data
          cells6="cells6_${generation}.txt"
          randcell6=$(cat "$cells6")
          randcell6=($randcell6)

          exits6="exits6_${generation}.txt"
          randexit6=$(cat "$exits6")
          randexit6=($randexit6)
        fi
      fi
    fi
  fi
fi

cd ..

if [ $batch_size -eq "1" ]
then
  if [ $guides -eq "1" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
  fi

  if [ $guides -eq "2" ]
  then
    #echo ${randcell1[$SIMULATION_ID]}
    #echo ${randexit1[$SIMULATION_ID]}
    #echo ${randcell2[$SIMULATION_ID]}
    #echo ${randexit2[$SIMULATION_ID]}
    #echo ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
  fi

  if [ $guides -eq "3" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
  fi

  if [ $guides -eq "4" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
  fi

  if [ $guides -eq "5" ]
  then    
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
  fi

  if [ $guides -eq "6" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${randcell6[$SIMULATION_ID]} ${randexit6[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
  fi
elif [ $batch_size -eq "2" ]
then
  if [ $guides -eq "1" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
  fi

  if [ $guides -eq "2" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
  fi

  if [ $guides -eq "3" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
  fi

  if [ $guides -eq "4" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
  fi

  if [ $guides -eq "5" ]
  then    
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${randcell5[$SIMULATION_ID+1]} ${randexit5[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
  fi

  if [ $guides -eq "6" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${randcell6[$SIMULATION_ID]} ${randexit6[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${randcell5[$SIMULATION_ID+1]} ${randexit5[$SIMULATION_ID+1]} ${randcell6[$SIMULATION_ID+1]} ${randexit6[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
  fi
elif [ $batch_size -eq "3" ]
then
  if [ $guides -eq "1" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
  fi

  if [ $guides -eq "2" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
  fi

  if [ $guides -eq "3" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
  fi

  if [ $guides -eq "4" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${randcell4[$SIMULATION_ID+2]} ${randexit4[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
  fi

  if [ $guides -eq "5" ]
  then    
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${randcell5[$SIMULATION_ID+1]} ${randexit5[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${randcell4[$SIMULATION_ID+2]} ${randexit4[$SIMULATION_ID+2]} ${randcell5[$SIMULATION_ID+2]} ${randexit5[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
  fi

  if [ $guides -eq "6" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${randcell6[$SIMULATION_ID]} ${randexit6[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${randcell5[$SIMULATION_ID+1]} ${randexit5[$SIMULATION_ID+1]} ${randcell6[$SIMULATION_ID+1]} ${randexit6[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${randcell4[$SIMULATION_ID+2]} ${randexit4[$SIMULATION_ID+2]} ${randcell5[$SIMULATION_ID+2]} ${randexit5[$SIMULATION_ID+2]} ${randcell6[$SIMULATION_ID+2]} ${randexit6[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
  fi
elif [ $batch_size -eq "4" ]
then
  if [ $guides -eq "1" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+3]} ${randexit1[$SIMULATION_ID+3]} ${scenarios[$SIMULATION_ID+3]}

  fi

  if [ $guides -eq "2" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+3]} ${randexit1[$SIMULATION_ID+3]} ${randcell2[$SIMULATION_ID+3]} ${randexit2[$SIMULATION_ID+3]} ${scenarios[$SIMULATION_ID+3]}
  fi

  if [ $guides -eq "3" ]
  then
    #echo ${randcell1[$SIMULATION_ID]}
    #echo ${randexit1[$SIMULATION_ID]}
    #echo ${randcell2[$SIMULATION_ID]}
    #echo ${randexit2[$SIMULATION_ID]}
    #echo ${randcell3[$SIMULATION_ID]}
    #echo ${randexit3[$SIMULATION_ID]}
    #echo ${scenarios[$SIMULATION_ID]}

    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}

    #echo ${randcell1[$SIMULATION_ID+1]}
    #echo ${randexit1[$SIMULATION_ID+1]}
    #echo ${randcell2[$SIMULATION_ID+1]}
    #echo ${randexit2[$SIMULATION_ID+1]}
    #echo ${randcell3[$SIMULATION_ID+1]}
    #echo ${randexit3[$SIMULATION_ID+1]}
    #echo ${scenarios[$SIMULATION_ID+1]}

    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}

    #echo ${randcell1[$SIMULATION_ID+2]}
    #echo ${randexit1[$SIMULATION_ID+2]}
    #echo ${randcell2[$SIMULATION_ID+2]}
    #echo ${randexit2[$SIMULATION_ID+2]}
    #echo ${randcell3[$SIMULATION_ID+2]}
    #echo ${randexit3[$SIMULATION_ID+2]}
    #echo ${scenarios[$SIMULATION_ID+2]}

    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}

    #echo ${randcell1[$SIMULATION_ID+3]}
    #echo ${randexit1[$SIMULATION_ID+3]}
    #echo ${randcell2[$SIMULATION_ID+3]}
    #echo ${randexit2[$SIMULATION_ID+3]}
    #echo ${randcell3[$SIMULATION_ID+3]}
    #echo ${randexit3[$SIMULATION_ID+3]}
    #echo ${scenarios[$SIMULATION_ID+3]}

    python shell_run_complex.py ${randcell1[$SIMULATION_ID+3]} ${randexit1[$SIMULATION_ID+3]} ${randcell2[$SIMULATION_ID+3]} ${randexit2[$SIMULATION_ID+3]} ${randcell3[$SIMULATION_ID+3]} ${randexit3[$SIMULATION_ID+3]} ${scenarios[$SIMULATION_ID+3]}
  fi

  if [ $guides -eq "4" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${randcell4[$SIMULATION_ID+2]} ${randexit4[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+3]} ${randexit1[$SIMULATION_ID+3]} ${randcell2[$SIMULATION_ID+3]} ${randexit2[$SIMULATION_ID+3]} ${randcell3[$SIMULATION_ID+3]} ${randexit3[$SIMULATION_ID+3]} ${randcell4[$SIMULATION_ID+3]} ${randexit4[$SIMULATION_ID+3]} ${scenarios[$SIMULATION_ID+3]}
  fi

  if [ $guides -eq "5" ]
  then    
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${randcell5[$SIMULATION_ID+1]} ${randexit5[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${randcell4[$SIMULATION_ID+2]} ${randexit4[$SIMULATION_ID+2]} ${randcell5[$SIMULATION_ID+2]} ${randexit5[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+3]} ${randexit1[$SIMULATION_ID+3]} ${randcell2[$SIMULATION_ID+3]} ${randexit2[$SIMULATION_ID+3]} ${randcell3[$SIMULATION_ID+3]} ${randexit3[$SIMULATION_ID+3]} ${randcell4[$SIMULATION_ID+3]} ${randexit4[$SIMULATION_ID+3]} ${randcell5[$SIMULATION_ID+3]} ${randexit5[$SIMULATION_ID+3]} ${scenarios[$SIMULATION_ID+3]}
  fi

  if [ $guides -eq "6" ]
  then
    python shell_run_complex.py ${randcell1[$SIMULATION_ID]} ${randexit1[$SIMULATION_ID]} ${randcell2[$SIMULATION_ID]} ${randexit2[$SIMULATION_ID]} ${randcell3[$SIMULATION_ID]} ${randexit3[$SIMULATION_ID]} ${randcell4[$SIMULATION_ID]} ${randexit4[$SIMULATION_ID]} ${randcell5[$SIMULATION_ID]} ${randexit5[$SIMULATION_ID]} ${randcell6[$SIMULATION_ID]} ${randexit6[$SIMULATION_ID]} ${scenarios[$SIMULATION_ID]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+1]} ${randexit1[$SIMULATION_ID+1]} ${randcell2[$SIMULATION_ID+1]} ${randexit2[$SIMULATION_ID+1]} ${randcell3[$SIMULATION_ID+1]} ${randexit3[$SIMULATION_ID+1]} ${randcell4[$SIMULATION_ID+1]} ${randexit4[$SIMULATION_ID+1]} ${randcell5[$SIMULATION_ID+1]} ${randexit5[$SIMULATION_ID+1]} ${randcell6[$SIMULATION_ID+1]} ${randexit6[$SIMULATION_ID+1]} ${scenarios[$SIMULATION_ID+1]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+2]} ${randexit1[$SIMULATION_ID+2]} ${randcell2[$SIMULATION_ID+2]} ${randexit2[$SIMULATION_ID+2]} ${randcell3[$SIMULATION_ID+2]} ${randexit3[$SIMULATION_ID+2]} ${randcell4[$SIMULATION_ID+2]} ${randexit4[$SIMULATION_ID+2]} ${randcell5[$SIMULATION_ID+2]} ${randexit5[$SIMULATION_ID+2]} ${randcell6[$SIMULATION_ID+2]} ${randexit6[$SIMULATION_ID+2]} ${scenarios[$SIMULATION_ID+2]}
    python shell_run_complex.py ${randcell1[$SIMULATION_ID+3]} ${randexit1[$SIMULATION_ID+3]} ${randcell2[$SIMULATION_ID+3]} ${randexit2[$SIMULATION_ID+3]} ${randcell3[$SIMULATION_ID+3]} ${randexit3[$SIMULATION_ID+3]} ${randcell4[$SIMULATION_ID+3]} ${randexit4[$SIMULATION_ID+3]} ${randcell5[$SIMULATION_ID+3]} ${randexit5[$SIMULATION_ID+3]} ${randcell6[$SIMULATION_ID+3]} ${randexit6[$SIMULATION_ID+3]} ${scenarios[$SIMULATION_ID+3]}
  fi
fi
