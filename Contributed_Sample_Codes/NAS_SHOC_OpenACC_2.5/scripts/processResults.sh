#!/bin/bash

USAGE="processResults.sh <conf_file>"

if [ "$#" -ne "1" ]; then
    echo $USAGE
    exit -1
fi

# folder exists
CONF_FILE=$1

# function for loggin to stderr
function log() {
    if [ "$#" -gt "0" ]; then
        (>&2 echo -en "\033[31m")  ## red
        (>&2 echo $@) # redirecting echo to stderr (file descriptor 2)
        (>&2 echo -en "\033[0m")  ## reset color
    fi
}

# function for processing folder
function processFolder() {
    BENCHMARK=$1
    FOLDER=$2
    log Processing result folder \'$FOLDER\'...
    
    for file in `ls $FOLDER | grep '\<res.*.txt\>'`; do
        log Processing $file...
        RES_ARRAY=("'$BENCHMARK'")
        python pcc-0.1.0/parser.py $FOLDER/$file > .tmp_processing
        while IFS='' read -r line || [[ -n "$line" ]]; do
            # Allows simple use of comments.
            if [ "${#line}" -gt "0" -a ! "${line:0:1}" == "#" ]; then
                #echo "'${line}'"
            
                # general configuration not tie-up to ACC or OMP
                if [ "${line:0:5}" == "Class" ]; then
                    val=""
                    for ith in `seq 0 ${#line}`; do
                        letter=${line:$ith:1}
                    
                        if [ "$letter" == ":" ]; then
                            val=${line:$[ith+1]}
                            break
                        fi
                    done
                    RES_ARRAY[1]=$val
                elif [ "${line:0:8}" == "ExecTime" ]; then
                    val=""
                    for ith in `seq 0 ${#line}`; do
                        letter=${line:$ith:1}
                    
                        if [ "$letter" == ":" ]; then
                            val=${line:$[ith+1]}
                            break
                        fi
                    done
                    RES_ARRAY[2]=$val
                fi
            fi
        done < ".tmp_processing"
        echo ${RES_ARRAY[0]}:${RES_ARRAY[1]}:${RES_ARRAY[2]}
    done
}

# processing execution time results
log Processing conf file

while IFS='' read -r line || [[ -n "$line" ]]; do
    # Allows simple use of comments.
    if [ "${#line}" -gt "0" -a ! "${line:0:1}" == "#" ]; then
        log "'${line}'"

        # processing result folder
        if [ "${line:0:1}" == ":" ]; then
            for ith in `seq 1 ${#line}`; do
                letter=${line:$ith:1}

                if [ "$letter" == ":" ]; then
                    break 
                fi
            done
            benchmark=${line:1:$[ith-1]}
            result_folder=`eval echo ${line:$[ith+1]}`
           
            if [ ! -d $result_folder ]; then 
                log ERROR: Skipping result folder \'$result_folder\' due that it is not a folder.; 
            else
                processFolder "$benchmark" "$result_folder"
            fi
        else
            eval ${line}
        fi
    fi
done < "$CONF_FILE"


