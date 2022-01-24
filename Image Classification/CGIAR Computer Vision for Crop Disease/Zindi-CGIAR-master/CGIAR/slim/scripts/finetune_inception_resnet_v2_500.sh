#!/bin/bash
##
# 2020, Lukas Picek, PiVa AI / University of West Bohemia
##
# The script is expected to run on server.
# Contains only training, not building tfrecord data files.
##

# Set for your visible device
export CUDA_VISIBLE_DEVICES=3

# Path to the tf_recorts
TF_DATASET_DIR=/mnt/datagrid/personal/picekluk/CGIAR/tfrecords_trainval

# Path where the checkpoints are stored
TF_TRAIN_DIR=/mnt/datagrid/personal/picekluk/CGIAR/checkpoints/inception_resnet_v2_plantclef_500_trainval_test_for_zindi/

# Path to the pretrained Checkpoint -> DOWNLOAD FROM http://ptak.felk.cvut.cz/personal/sulcmila/models/LifeCLEF2019/
TF_CHECKPOINT_PATH=/mnt/datagrid/personal/picekluk/pretrained_models/PlantCLEF2018/inception_resnet_v2/model.ckpt-1200000

mkdir -p $TF_TRAIN_DIR

python train_image_classifier.py \
    --train_dir=${TF_TRAIN_DIR} \
    --dataset_dir=${TF_DATASET_DIR} \
    --dataset_name=CGIAR \
    --dataset_split_name=train \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=${TF_CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits/Logits,InceptionResnetV2/AuxLogits/Conv2d_2a_5x5 \
    --ignore_missing_vars=True \
    --train_image_size=500 \
    --max_number_of_steps=18000 \
    --save_interval_steps=1000 \
    --save_interval_secs=1800 \
    --save_summaries_secs=1800 \
    --moving_average_decay=0.999 \
    --modest=True \
    --batch_size=16
