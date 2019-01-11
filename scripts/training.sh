#!/usr/bin/env bash

OPEN_NMT_PATH=/home/lvapeab/smt/software/openNMT-py

# Task and data options
TASK=ue
SRC_LAN=en
TRG_LAN=de
DATA_PATH=/home/lvapeab/DATASETS/${TASK}/${SRC_LAN}${TRG_LAN}/joint_bpe/
TRAIN_FILENAME=training
DEV_FILENAME=dev
DATASET_NAME=$DATA_PATH/open_nmt/Dataset_${TASK}_${SRC_LAN}${TRG_LAN}
DATA=${DATASET_NAME}

# Model options
MODEL_TYPE="text"
SRC_WORD_VEC_SIZE=512
TGT_WORD_VEC_SIZE=512
WORD_VEC_SIZE=512
SHARE_EMBEDDINGS="" #--share_decoder_embeddings
POSITION_ENCODING="--position_encoding"
FEAT_VEC_SIZE=512
ENCODER_TYPE="transformer"
DECODER_TYPE="transformer"
LAYERS=4
ENC_LAYERS=4
DEC_LAYERS=4
RNN_SIZE=512
ENC_RNN_SIZE=512
DEC_RNN_SIZE=512
INPUT_FEED=1
RNN_TYPE=LSTM
GLOBAL_ATTENTION="mlp"
GLOBAL_ATTENTION_FUNCTION="softmax"
SELF_ATTN_TYPE="scaled-dot"
HEADS=8
TRANSFORMER_FF=2048
GENERATOR_FUNCTION="softmax"

# Optimization options
BATCH_SIZE=50
BATCH_TYPE="sents"
ACCUM_COUNT=1
VALID_STEPS=3750
VALID_BATCH_SIZE=32
MAX_GENERATOR_BATCHES=32
TRAIN_STEPS=120000
OPTIM="adam"
LEARNING_RATE="0.0002"
LEARNING_RATE_DECAY=1
START_DECAY_STEPS=1
DECAY_STEPS=1
DECAY_METHOD=none
WARMUP_STEPS=4000
MAX_GRAD_NORM=5
LABEL_SMOOTHING=0.1
DROPOUT=0.1

MODEL_PATH=/home/lvapeab/MODELS/${TASK}/OpenNMT/${SRC_LAN}${TRG_LAN}_${ENCODER_TYPE}_${DECODER_TYPE}_${SRC_WORD_VEC_SIZE}_${TGT_WORD_VEC_SIZE}_${ENC_LAYERS}_${DEC_LAYERS}

# Logging options
REPORT_EVERY=1
LOG_FILE=$OPEN_NMT_PATH/logs/${TASK}_${SRC_LAN}${TRG_LAN}_${ENCODER_TYPE}_${DECODER_TYPE}
LOG_FILE_LEVEL=WARNING
EXP=${TASK}_${SRC_LAN}${TRG_LAN}
TENSORBOARD="--tensorboard"
TENSORBOARD_LOG_DIR=$MODEL_PATH/tensorboard/

SAVE_MODEL="model"
SAVE_CHECKPOINT_STEPS=3750

mkdir -p `dirname ${LOG_FILE}`

python ${OPEN_NMT_PATH}/train.py \
                -src_word_vec_size ${SRC_WORD_VEC_SIZE} \
                -tgt_word_vec_size ${TGT_WORD_VEC_SIZE} \
                -word_vec_size ${WORD_VEC_SIZE} \
                -model_type ${MODEL_TYPE} \
                -encoder_type ${ENCODER_TYPE} \
                -decoder_type ${DECODER_TYPE} \
                -layers ${LAYERS} \
                -enc_layers ${ENC_LAYERS} \
                -dec_layers ${DEC_LAYERS} \
                -rnn_size ${RNN_SIZE} \
                -enc_rnn_size ${ENC_RNN_SIZE} \
                -dec_rnn_size ${DEC_RNN_SIZE} \
                -input_feed ${INPUT_FEED} \
                -rnn_type ${RNN_TYPE} \
                -global_attention ${GLOBAL_ATTENTION} \
                -global_attention_function ${GLOBAL_ATTENTION_FUNCTION} \
                -self_attn_type ${SELF_ATTN_TYPE} \
                -heads ${HEADS} \
                -transformer_ff ${TRANSFORMER_FF} \
                -generator_function ${GENERATOR_FUNCTION} \
                ${SHARE_EMBEDDINGS} \
                ${POSITION_ENCODING} \
                -data ${DATA} \
                -save_model ${SAVE_MODEL} \
                -save_checkpoint_steps ${SAVE_CHECKPOINT_STEPS} \
                -batch_size ${BATCH_SIZE} \
                -batch_type ${BATCH_TYPE} \
                -train_steps ${TRAIN_STEPS} \
                -optim ${OPTIM} \
                -max_grad_norm ${MAX_GRAD_NORM} \
                -dropout ${DROPOUT} \
                -label_smoothing ${LABEL_SMOOTHING} \
                -learning_rate ${LEARNING_RATE} \
                -learning_rate_decay ${LEARNING_RATE_DECAY} \
                -start_decay_steps ${START_DECAY_STEPS} \
                -decay_steps ${DECAY_STEPS} \
                -decay_method ${DECAY_METHOD} \
                -warmup_steps ${WARMUP_STEPS} \
                -report_every ${REPORT_EVERY} \
                -log_file ${LOG_FILE} \
                -exp ${EXP} \
                 ${TENSORBOARD} \
                -tensorboard_log_dir ${TENSORBOARD_LOG_DIR} \
                -world_size 1 \
                -gpu_ranks 0
