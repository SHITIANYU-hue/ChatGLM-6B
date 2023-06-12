PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python3 web_demolora.py \
    --model_name_or_path THUDM/chatglm-6b \
    --lora_checkpoint /home/tianyushi/code/ChatGLM-Tuning/output/checkpoint-51000 \
    --pre_seq_len $PRE_SEQ_LEN
