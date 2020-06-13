#!/bin/bash

population=$1
n_scenarios=$2
MAXCOUNT=$(($population*$n_scenarios))

count=1
scenario=0
while [ "$count" -le $MAXCOUNT ]; do
  scenarios[$count]=$scenario
  let "scenario += 1"
  echo $scenario
  if [ "$scenario" -eq $n_scenarios ]
  then
    scenario=0
  fi
  let "count += 1"
done

echo ${scenarios[*]} >> scenarios.txt
