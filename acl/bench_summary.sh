#!/bin/bash

# 定义预训练模型列表
#pretrained_models=(
#    "ahxt/LiteLlama-460M-1T"
#    "Qwen/Qwen2-0.5B-Instruct"
#    "bigscience/bloom-560m"
#    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#    "EleutherAI/gpt-neo-1.3B"
#    "Qwen/Qwen2-1.5B-Instruct"
#    "internlm/internlm2-1_8b"
#    "internlm/internlm2-chat-1_8b-sft"
#    "openbmb/MiniCPM-2B-sft-bf16"
#    "microsoft/phi-2"
#    "microsoft/Phi-3-mini-4k-instruct"
#    "THUDM/chatglm3-6b"
#    "mistralai/Mistral-7B-v0.1"
#    "mistralai/Mistral-7B-Instruct-v0.2"
#
#)
#special process
#chatglm3-6b
#pretrained_models=( "google/gemma-1.1-2b-it")
pretrained_models=( "microsoft/Phi-3-mini-4k-instruct")
#pretrained_models=( "Qwen/Qwen2-7B-Instruct" "Qwen/Qwen2-7B")

# 循环执行命令
for model in "${pretrained_models[@]}"; do
    # 提取预训练模型名称后缀
    #model_suffix="${model#*/}"
    model_suffix="${model//\//_}"
    echo "$model_suffix"
    # 构建output_path参数
    output_path1="./qwen1_5/${model_suffix}_new_version"
    output_path2="./llama2/${model_suffix}_new_version"
    
    # 执行lm_eval命令
    #qwen1.5 reference
    lm_eval --model hf \
            --model_args "pretrained=$model",trust_remote_code=True \
            --tasks cnndm_qwen_no_limit_len,xsum_qwen_no_limit_len,newsroom_qwen_no_limit_len,bbc2024_qwen_no_limit_len \
            --device cuda:2 \
            --batch_size auto\
            --num_fewshot 0 \
            --log_samples \
            --output_path "$output_path1"
    lm_eval --model hf \
            --model_args "pretrained=$model",trust_remote_code=True \
            --tasks cnndm_llama2_no_limit_len,xsum_llama2_no_limit_len,newsroom_llama2_no_limit_len,bbc2024_llama2_no_limit_len \
            --device cuda:2 \
            --batch_size auto \
            --num_fewshot 0 \
            --log_samples \
            --output_path "$output_path2" 
done

