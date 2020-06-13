#!/bin/bash

cells=(20 21 22 65 66 67 110 111 112 155 156 157 200 201 202 245 246 247 290 291 292 335 336 337 560 561 562 605 606 607 650 651 652 695 696 697 740 741 742 785 786 787 830 831 832 875 876 900 901 902 903 904 905 906 907 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 937 938 939 940 941 942 945 946 947 948 949 950 951 952 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 982 983 984 985 986 987 990 991 992 993 994 995 996 997 1002 1003 1004 1005 1006 1007 1008 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1028 1029 1030 1031 1032 1055 1056 1057 1100 1101 1102 1145 1146 1147 1190 1191 1192 1235 1236 1237 1280 1281 1282 1325 1326 1327 1370 1371 1372 1415 1416 1417 1460 1461 1685 1686 1687 1730 1731 1732 1775 1776 1777 1820 1821 1822 1865 1866 1867 1910 1911)
exits=(0 1 2 3)
ncells=${#cells[@]}
nexits=${#exits[@]}
population=$1
scenarios=$2

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell1[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit1[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${#randcell1[@]}
echo ${#randexit1[@]}

echo ${randcell1[*]} >> "cells1_0.txt"
echo ${randexit1[*]} >> "exits1_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell2[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit2[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell2[*]} >> "cells2_0.txt"
echo ${randexit2[*]} >> "exits2_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell3[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit3[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell3[*]} >> "cells3_0.txt"
echo ${randexit3[*]} >> "exits3_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell4[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit4[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell4[*]} >> "cells4_0.txt"
echo ${randexit4[*]} >> "exits4_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell5[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit5[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell5[*]} >> "cells5_0.txt"
echo ${randexit5[*]} >> "exits5_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell6[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit6[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell6[*]} >> "cells6_0.txt"
echo ${randexit6[*]} >> "exits6_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell7[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit7[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell7[*]} >> "cells7_0.txt"
echo ${randexit7[*]} >> "exits7_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell8[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit8[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell8[*]} >> "cells8_0.txt"
echo ${randexit8[*]} >> "exits8_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell9[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit9[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell9[*]} >> "cells9_0.txt"
echo ${randexit9[*]} >> "exits9_0.txt"

count=1
while [ "$count" -le $population ]; do
  rnd=${cells[$((RANDOM%$ncells))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randcell10[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

count=1
while [ "$count" -le $population ]; do
  rnd=${exits[$((RANDOM%$nexits))]}
  b_count=1
  while [ "$b_count" -le $scenarios ]; do
    indx=$((($count-1)*$scenarios + $b_count))
    randexit10[$indx]=$rnd
    let "b_count += 1"
  done
  let "count += 1"
done

echo ${randcell10[*]} >> "cells10_0.txt"
echo ${randexit10[*]} >> "exits10_0.txt"
