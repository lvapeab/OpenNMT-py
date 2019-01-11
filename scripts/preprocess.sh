#!/usr/bin/env bash

OPEN_NMT_PATH=/home/lvapeab/smt/software/openNMT-py

TASK=ue
SRC_LAN=en
TRG_LAN=de
DATA_PATH=/home/lvapeab/DATASETS/${TASK}/${SRC_LAN}${TRG_LAN}/joint_bpe/
TRAIN_FILENAME=training
DEV_FILENAME=dev
DEST=$DATA_PATH/open_nmt
DATASET_NAME=Dataset_${TASK}_${SRC_LAN}${TRG_LAN}

python ${OPEN_NMT_PATH}/preprocess.py --train_src ${DATA_PATH}/${TRAIN_FILENAME}.${SRC_LAN} \
                     --train_tgt ${DATA_PATH}/${TRAIN_FILENAME}.${TRG_LAN} \
                     --valid_src ${DATA_PATH}/${DEV_FILENAME}.${SRC_LAN} \
                     --valid_tgt ${DATA_PATH}/${DEV_FILENAME}.${TRG_LAN} \
                     --src_seq_length_trunc 50 --tgt_seq_length_trunc 50 \
                     --save_data ${DEST}/${DATASET_NAME}

