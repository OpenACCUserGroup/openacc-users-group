#!/bin/bash
# author sergiop@udel.edu


USAGE="./singleExperiment.sh \"CLASS_LETTER[ CLASS_LETTER...]\" <benchmark_suite> <threads_env_variable> <number of threads> . E.g. ./singleExperiment.sh \"A B C\" nas_acc 4 "

if [ "$#" -lt "2" ]; then
    echo $USAGE
    exit -1
fi

# defining functions
function run_nas() {

    INPUT1=$1
    CLASSES=(${INPUT1[@]})
    BENCHMARK_SUITE=$2

    echo "Executing NAS benchmark" | $LOG
    
    echo Compiling again... | $LOG
    cd ../sys
    make clean | $LOG
    
    cd $BENCHMARK_FOLDER
    make clean | $LOG
    
    if [ "$BENCHMARK_SUITE" == "nas_cuda" ]; then
        echo "Executing CUDA benchmark" | $LOG
        echo "make $TEST" | $LOG
        (make $TEST) 2>&1 | tee -a $BASE_FOLDER/$FOLDER/compile.txt 
    else
        for CLASS in ${CLASSES[@]}; do
            echo "make CC=$CC CLASS=$CLASS DEFINES="$DEFINES" $TEST" | $LOG
            (make CC=$CC CLASS=$CLASS DEFINES="$DEFINES" $TEST) 2>&1 | tee -a $BASE_FOLDER/$FOLDER/compile$CLASS.txt 
        done
    fi
    
    echo Running experiments... $(date +%m%d%y_%H%M%S) | $LOG
    
    for CLASS in ${CLASSES[@]}; do
        if [ "$RECORD_CPU" == "1" ]; then 
            $RECORD_APP $BENCHMARK_EXEC.$CLASS.x $BASE_FOLDER/$FOLDER/cpu$CLASS.txt&
        fi
        
        if [ "$BENCHMARK_SUITE" == "nas_cuda" ]; then
            echo "./$BENCHMARK_EXEC $CLASS" | $LOG
            (time ./$BENCHMARK_EXEC $CLASS) 2>&1 | tee -a $BASE_FOLDER/$FOLDER/res$CLASS.txt 
        else
            echo "./$BENCHMARK_EXEC.$CLASS.x" | $LOG
            (time ./$BENCHMARK_EXEC.$CLASS.x) 2>&1 | tee -a $BASE_FOLDER/$FOLDER/res$CLASS.txt 
        fi
        
        if [ "$RECORD_CPU" == "1" ]; then echo "1" > .cpu_test; kill -9 $!; fi
    done
}

function run_shoc() {
    
    INPUT1=$1
    CLASSES=(${INPUT1[@]})
    BENCHMARK_SUITE=$2
    echo "Executing SHOC benchmark" | $LOG
    
    MAKE_FILE=""
    if [ "$PXM" == "acc" ]; then
        MAKE_FILE=Makefile.acc
    elif [ "$PXM" == "omp" ]; then
        MAKE_FILE=Makefile.omp
    else
        echo "ERROR: PXM should be assigned to either acc or omp. Use 'export PXM=acc' in your Configuration file."
        exit -1
    fi
    
    echo Compiling again... | $LOG
    cd $BENCHMARK_FOLDER
    make -f $MAKE_FILE clean | $LOG

    echo "make -f $MAKE_FILE CC=$CC TA=$TA DEFINES="$DEFINES" $TEST" | $LOG
    (make -f $MAKE_FILE CC=$CC TA=$TA DEFINES="$DEFINES" $TEST) 2>&1 | tee -a $BASE_FOLDER/$FOLDER/compile.txt 
    
    echo Running experiments... $(date +%m%d%y_%H%M%S) | $LOG
    
    for CLASS in ${CLASSES[@]}; do
        if [ "$RECORD_CPU" == "1" ]; then
            $RECORD_APP $BENCHMARK_EXEC $BASE_FOLDER/$FOLDER/cpu$CLASS.txt&
        fi
        
        echo "./$BENCHMARK_EXEC -s $CLASS $BENCHMARK_EXEC_ARGS" | $LOG 
        (time ./$BENCHMARK_EXEC -s $CLASS $BENCHMARK_EXEC_ARGS) 2>&1 | tee -a $BASE_FOLDER/$FOLDER/res$CLASS.txt 
        
        if [ "$RECORD_CPU" == "1" ]; then echo "1" > .cpu_test; kill -9 $!; fi
    done
}

# getting the arguments
#INPUT1=$1
#CLASSES=(${INPUT1[@]})
CLASSES=$1
BENCHMARK_SUITE=$2

# running on multicore
if [ "$#" -ge "3" ]; then
    ENV_THREADS=$3
    NUM_THREADS=$4

    export $ENV_THREADS=$NUM_THREADS
    echo Running multicore with $ENV_THREADS=$NUM_THREADS

    RECORD_CPU=1
    RECORD_APP=$PWD/record_cpu.sh
fi

EXEC_DATE=$(date +%m%d%y_%H%M%S)

BASE_FOLDER=$EXPERIMENTS_HOME/$BENCHMARK_NAME

if [ -n "$NUM_THREADS" ]; then
    FOLDER=$BENCHMARK_NAME"_nth"$NUM_THREADS"_"$EXEC_DATE
else
    FOLDER=$BENCHMARK_NAME"_"$EXEC_DATE
fi

echo creating exp folder $BASE_FOLDER/$FOLDER

mkdir -p $BASE_FOLDER/$FOLDER

LOG="tee -a $BASE_FOLDER/$FOLDER/log.txt"

echo Executing experiment on $EXEC_DATE... | $LOG
echo Configuration | $LOG
echo --------------------------------  | $LOG
echo CC=$CC | $LOG
echo ACCEL_INFO=$ACCEL_INFO | $LOG
echo CLASSES=${CLASSES[@]} | $LOG
echo BENCHMARK_NAME=$BENCHMARK_NAME | $LOG
echo BENCHMARK_FOLDER=$BENCHMARK_FOLDER | $LOG
echo BENCHMARK_EXEC=$BENCHMARK_EXEC | $LOG
echo BENCHMARK_EXEC_ARGS=$BENCHMARK_EXEC_ARGS | $LOG
echo BENCHMARK_SUITE=$BENCHMARK_SUITE | $LOG
echo PXM=$PXM | $LOG
echo RESULTS_FOLDER=$BASE_FOLDER/$FOLDER | $LOG
echo EXPERIMENTS_HOME=$EXPERIMENTS_HOME | $LOG
echo DEFINES=$DEFINES | $LOG
echo EXTRAS=$EXTRAS | $LOG
echo -------------------------------- | $LOG

if [ -z "$EXPERIMENTS_HOME" ]; then
    echo "Setting EXPERIMENTS_HOME to pwd" | $LOG
    EXPERIMENTS_HOME=$PWD
fi

echo saving node info | $LOG
(echo ACCELERATOR; $ACCEL_INFO; echo && echo CPU; lscpu; echo && echo HOSTNAME; hostname) 2>&1 | tee -a $BASE_FOLDER/$FOLDER/node_info.txt

echo CDing into benchmark folder \'$BENCHMARK_FOLDER\' | $LOG
cd $BENCHMARK_FOLDER

# Executing NAS benchmarks
if [ "$BENCHMARK_SUITE" == "nas_cuda" -o "$BENCHMARK_SUITE" == "nas_acc" -o "$BENCHMARK_SUITE" == "nas_omp" ]; then
    run_nas "$CLASSES" $BENCHMARK_SUITE;
# Executing SHOC benchmarks
elif [ "$BENCHMARK_SUITE" == "shoc" ]; then
    run_shoc "$CLASSES" $BENCHMARK_SUITE;
else
    echo "ERROR: Unrecognized benchmark suite. Possible values: \"nas_cuda\", \"nas_acc\", \"nas_omp\", \"shoc\""
    exit -1
fi    

echo Finishing experiments... $(date +%m%d%y_%H%M%S) | $LOG
   
