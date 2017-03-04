#!/bin/bash
# author: sergiop@udel.edu

if [ ! "$#" -eq "2" ]; then
    echo "./record_cpu.sh <APP> <LOGFILE>"
    exit -1
fi

APP=$1
LOG=$2

rm .cpu_test
touch .cpu_test

until [ "`cat .cpu_test | wc -l`" -gt "0" ]; do
    top -d 1 -n 5 -b | grep $APP >> $LOG
done
