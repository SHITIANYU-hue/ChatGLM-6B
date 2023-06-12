PRE_SEQ_LEN=128
LR=1e-2
CHAT_TRAIN_DATA='/home/tianyushi/code/ChatGLM-6B/ptuning/character/trainkeai.json'
CHECKPOINT_NAME='chatglm-keai'
CHAT_VAL_DATA='/home/tianyushi/code/ChatGLM-6B/ptuning/character/trainkeai.json'
PTCHECKPOINT_NAME='/home/tianyushi/code/ChatGLM-6B/ptuning/chatglm-gangjingmulti/checkpoint-3000'


CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file $CHAT_TRAIN_DATA \
    --validation_file $CHAT_VAL_DATA \
    --prompt_column instruction \
    --response_column output \
    --history_column input \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir $CHECKPOINT_NAME \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

