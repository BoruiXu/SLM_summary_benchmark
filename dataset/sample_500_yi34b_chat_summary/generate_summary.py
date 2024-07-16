#using Qwen generate summary

from datasets import load_dataset

import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import json
# Set OpenAI's API key and API base to use vLLM's API server.
from scipy.stats import kendalltau


prompt = "You are a helpful summary assistant. You can help users summarize news in two sentences." #two sentences


from openai import OpenAI          
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8880/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

#load dataset
from datasets import load_dataset

#/home/xbr/LLM/dataset/summary/sample_dataset
#cnndm_sample_0k5_1k5.json  newsroom_sample_0k5_1k5.json  xsum_sample_0k5_1k5.json

def clean_summary(summary):
     
    summary = summary.strip()
    summary = summary.replace('The news summary is: \"', '')
    # summary = summary.split('\n')[0]
    summary_list = summary.split('\n')
    if(len(summary_list)<3):
        summary = summary_list[-1]
    else:
        summary = summary.split('\n')[2]
    candidate = summary
    complete_sentences = re.findall(r'[^.!?]*[.!?]', summary)
    # 将匹配到的完整句子连接成一个新的字符串
    summary = ''.join(complete_sentences)
    if(summary==''):
        summary = candidate
    
    return summary
    

#generate qwen summary
# data = json.load(open('/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_cnndm_average.json'))
# data = json.load(open('/home/xbr/LLM/summary_benchmark/dataset/sample_500/cnndm_sample_500_0k5_1k5.jsonl'))
files = ['cnndm','newsroom','xsum','bbc2024']
for f in files:
    data = json.load(open('/home/xbr/LLM/summary_benchmark/dataset/sample_500/'+f+'_sample_500_0k5_1k5.jsonl'))
    
    for i in tqdm(range(len(data))):
        news = data[i]['article']
        
        chat_response = client.chat.completions.create(
            model="01-ai/Yi-34B-Chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please summarize the news in two sentences.\nNews: "+news+'\nSummary: '}, #two sentences
                
            ],
            temperature=0,
            # max_tokens=40,
        )
        
        res = chat_response.choices[0].message.content
        res = clean_summary(res)
        print(res)    
        data[i]['qwen_reference_summary'] = res
        
    #save to json
    #./filter_annotations_summeval_llama2_summary.jsonl
    #/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_cnndm_average_with_yi34b.json
    # with open('./data/likert_evaluation_results_cnndm_average_with_qwen32b.jsonl', 'w') as f:
    with open(f+'_sample_500_0k5_1k5_yi34b_chat_summary_no_len_limit.jsonl', 'w') as f:
        json.dump(data, f, indent=4)
