#!/bin/bash

echo "START PREPROCESS --->"
python run_preprocess.py --config_name config.yaml
echo "<--- END PREPROCESS"

echo "START TRAIN --->"
for i in `seq 0 4`
do
    echo "START - FOLD: $i"
    python run_train.py --config_name config.yaml --fold $i

    ret=$?
    if [ $ret -ne 0 ]; then
        echo "RAISED EXCEPTION"
        exit 1
    fi
    echo "END - FOLD: $i"
done
echo "<-- END TRAIN"
