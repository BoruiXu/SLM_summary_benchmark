#!/bin/bash

# 定义预训练模型列表
#pretrained_models=(
#    "ahxt/LiteLlama-460M-1T"
#    "Qwen/Qwen2-0.5B-Instruct"
#    "Qwen/Qwen2-0.5B"
#    "bigscience/bloom-560m"
#    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#    "EleutherAI/gpt-neo-1.3B"
#    "EleutherAI/gpt-neo-2.7B"
#    "Qwen/Qwen2-1.5B-Instruct"
#    "internlm/internlm2-1_8b"
#    "internlm/internlm2-chat-1_8b-sft"
#    "openbmb/MiniCPM-2B-sft-bf16"
#
#)
#special process
#chatglm3-6b
#pretrained_models=( "google/gemma-1.1-2b-it")
#pretrained_models=( "Qwen/Qwen1.5-4B" "Qwen/Qwen1.5-4B-Chat" "Qwen/Qwen1.5-0.5B" "Qwen/Qwen1.5-0.5B-Chat" "Qwen/Qwen1.5-1.8B" "Qwen/Qwen1.5-1.8B-Chat" "Qwen/Qwen1.5-7B" "Qwen/Qwen1.5-7B-Chat")
pretrained_models=("internlm/internlm2-1_8b"  )

# 循环执行命令
for model in "${pretrained_models[@]}"; do
    # 提取预训练模型名称后缀
    #model_suffix="${model#*/}"
    model_suffix="${model//\//_}"
    echo "$model_suffix"
    # 构建output_path参数
    output_path1="./qwen1_5/${model_suffix}_not_eos"
    output_path2="./llama2/${model_suffix}_not_eos"
    
    # 执行lm_eval命令
    #qwen1.5 reference
    #lm_eval --model hf \
    #        --model_args "pretrained=$model",trust_remote_code=True \
    #        --tasks cnndm_qwen_no_limit_len,xsum_qwen_no_limit_len,newsroom_qwen_no_limit_len,bbc2024_qwen_no_limit_len \
    #        --device cuda:3 \
    #        --batch_size auto \
    #        --num_fewshot 0 \
    #        --log_samples \
    #        --output_path "$output_path1"
    #lm_eval --model hf \
    #        --model_args "pretrained=$model",trust_remote_code=True \
    #        --tasks cnndm_llama2_no_limit_len,xsum_llama2_no_limit_len,newsroom_llama2_no_limit_len,bbc2024_llama2_no_limit_len \
    #        --device cuda:3 \
    #        --batch_size auto \
    #        --num_fewshot 0 \
    #        --log_samples \
    #        --output_path "$output_path2" 
    lm_eval --model hf \
            --model_args "pretrained=$model",trust_remote_code=True \
            --tasks bbc2024_llama2_no_limit_len_4_sentence,bbc2024_llama2_no_limit_len_3_sentence,bbc2024_llama2_no_limit_len_2_sentence,bbc2024_llama2_no_limit_len_1_sentence \
            --device cuda:7 \
            --batch_size 1 \
            --num_fewshot 0 \
            --log_samples \
            --output_path "./factKB_all/${model_suffix}" 
    #lm_eval --model hf \
    #        --model_args "pretrained=$model",trust_remote_code=True \
    #        --tasks cnndm_llama2_no_limit_len,xsum_llama2_no_limit_len,newsroom_llama2_no_limit_len,bbc2024_llama2_no_limit_len \
    #        --device cuda:7 \
    #        --batch_size 1 \
    #        --num_fewshot 0 \
    #        --log_samples \
    #        --output_path "./factKB_all/${model_suffix}" 
done

