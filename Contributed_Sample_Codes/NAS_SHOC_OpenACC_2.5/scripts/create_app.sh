#!/bin/bash

USAGE="./create_app.sh <orig_folder> <target_folder> <MACRO...>. MACR0=-Dkey=val"

if [ "$#" -lt "2" ]; then
    echo $USAGE
    exit -1
fi

CPP_PATH=$PWD/pcc-0.1.0

FOLDER_ORG=$1
FOLDER_TAR=$2

ARGS_SET=("${@:3:$#}")
echo Extra args: ${ARGS_SET[@]}

echo Working with Orig_folder: $FOLDER_ORG and Target_folder: $FOLDER_TAR

mkdir -p $FOLDER_TAR

# processing .c files
for file in `ls $FOLDER_ORG | grep '.c'`; do
    echo Processing $file
    python $CPP_PATH/cpp.py $FOLDER_ORG/$file $ARGS_SET 2&> $FOLDER_TAR/$file
done

# processing .h files
for file in `ls $FOLDER_ORG | grep '.h'`; do
    echo Processing $file
    python $CPP_PATH/cpp.py $FOLDER_ORG/$file $ARGS_SET 2&> $FOLDER_TAR/$file
done
