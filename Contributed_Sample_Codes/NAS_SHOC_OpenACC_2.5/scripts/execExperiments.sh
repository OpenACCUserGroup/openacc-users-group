#!/bin/bash

# author: sergiop@udel.edu

function USAGE() {
    echo "./execExperiments.sh <conf_file>."
    echo "For the configuration file:"
    echo "Variables formated as \"export MyVAR=value\":"
    echo "   EXPERIMENTS_HOME: folder where the results will be stored."
    echo "   CC: compiler to use."
    echo "   ACCEL_INFO: command to query the accelerator info."
    echo "   TEST: flags that will be passed to make."
    echo "   BENCHMARK_NAME: name of the benchmark folder to create."
    echo "   BENCHMARK_EXEC: prefix of the executed binary."
    echo "   BENCHMARK_FOLDER: folder where the source code (.c and .h) is located."
    echo "   DEFINES: string to be passed to make (assuming you have a DEFINES variable whithin your makefile)."
    echo "   TA: architecture to use for the gpu."
    echo "   PXM: Program execution model to use. E.g. PXM=acc, PXM=omp."
    echo "   EXTRA_CFLAGS: extra flags to pass to the c compiler."
    echo "   EXTRA_CLINKFLAGS: extra flags to pass to linker."
    echo "Variables formated as \"MyVAR=value\":"
    echo "   CLASSES: list of classes to use for NAS, e.g. CLASSES=A B C."
    echo "   THREADS: list of number of threads to use in multicore runs, e.g. THREADS=1 4."
    echo "   ENV_THREADS: env variable that sets the number of threads to use in the runtime. It depends if acc (ACC_NUM_CORES) or omp (OMP_NUM_THREADS)."
    echo "   BENCHMARK_SUITE: benchmark suite to used. Possible values: \"nas_cuda\", \"nas_acc\", \"nas_omp\", \"shoc\"."
}

if [ "$#" -ne "1" ]; then
    USAGE
    exit -1
fi

CONF_FILE=$1
CLASSES=()
THREADS=()
ENV_THREADS=""
FOLDER=""
BENCHMARK_SUITE=""

echo Loading conf file
while IFS='' read -r line || [[ -n "$line" ]]; do
    # Allows simple use of comments.
    if [ "${#line}" -gt "0" -a ! "${line:0:1}" == "#" ]; then
        echo "'${line}'"
        
        # general configuration not tie-up to ACC or OMP
        if [ "${line:0:7}" == "CLASSES" ]; then
            vals=""
            for ith in `seq 0 ${#line}`; do
                letter=${line:$ith:1}
                
                if [ "$letter" == "=" ]; then
                    vals=${line:$[ith+1]}
                    break
                fi
            done
            pos=0
            for val in $vals; do 
                CLASSES[$[pos++]]=$val
            done
        # multicore configurations
        elif [ "${line:0:7}" == "THREADS" ]; then
            vals=""
            for ith in `seq 0 ${#line}`; do
                letter=${line:$ith:1}
                
                if [ "$letter" == "=" ]; then
                    vals=${line:$[ith+1]}
                    break
                fi
            done
            pos=0
            for val in $vals; do 
                THREADS[$[pos++]]=$val
            done
        elif [ "${line:0:11}" == "ENV_THREADS" ]; then
            val=""
            for ith in `seq 0 ${#line}`; do
                letter=${line:$ith:1}
                
                if [ "$letter" == "=" ]; then
                    val=${line:$[ith+1]}
                    break
                fi
            done
            ENV_THREADS=$val
        elif [ "${line:0:15}" == "BENCHMARK_SUITE" ]; then
            val=""
            for ith in `seq 0 ${#line}`; do
                letter=${line:$ith:1}
                
                if [ "$letter" == "=" ]; then
                    val=${line:$[ith+1]}
                    break
                fi
            done
            BENCHMARK_SUITE=$val
        else
            eval ${line}
        fi
    fi
done < "$CONF_FILE"

if [ "$BENCHMARK_SUITE" != "nas_cuda" -a "$BENCHMARK_SUITE" != "nas_acc" -a "$BENCHMARK_SUITE" != "nas_omp" -a "$BENCHMARK_SUITE" != "shoc" ]; then
    echo "Error BENCHMARK_SUITE needs to be: \"nas_cuda\", \"nas_acc\", \"nas_omp\", \"shoc\"."
    exit -1
fi

if [ "$ENV_THREADS" == "" ]; then 
    # running non-multicore
    (./singleExperiment.sh "${CLASSES[*]}" ${BENCHMARK_SUITE}) 2>&1 | tee -a logExperiments.txt
else
    # running multicore
    # number of threads to use
    if [ "${#THREADS[@]}" -gt "0" ]; then 
        
        for th in ${THREADS[@]}; do
            export EXTRAS="ENV_THREADS=$ENV_THREADS":"NUM_THREADS=$th"
            #export $ENV_THREADS=$th
            (./singleExperiment.sh "${CLASSES[*]}" ${BENCHMARK_SUITE} $ENV_THREADS $th) 2>&1 | tee -a logExperiments.txt
        done
    else
        echo "ERROR: please set the THREADS option in your config file. e.g. THREADS=1 2 4"
    fi
fi


