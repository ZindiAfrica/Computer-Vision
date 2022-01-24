#!/bin/bash
##
# 2020, Lukas Picek, PiVa AI / University of West Bohemia
##
# The script is expected to run on server.
# Contains only training, not building tfrecord data files.
##

# Set for your visible device
export CUDA_VISIBLE_DEVICES=4

# Path to the tf_recorts
TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/CGIAR/tfrecords_trainval/

# Path where the checkpoints are stored
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/CGIAR/checkpoints/inception_v4_from_PlantClef2018_500_train_for_zindi/

# Path to the pretrained Checkpoint -> DOWNLOAD FROM http://ptak.felk.cvut.cz/personal/sulcmila/models/LifeCLEF2018/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/PlantCLEF2018/inception_v4/model.ckpt-2060000

mkdir -p $TF_TRAIN_DIR

python train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=CGIAR \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --ignore_missing_vars=True \
    --train_image_size=500 \
    --max_number_of_steps=20000 \
    --save_interval_steps=1000 \
    --save_interval_secs=1800 \
    --save_summaries_secs=1800 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=16

