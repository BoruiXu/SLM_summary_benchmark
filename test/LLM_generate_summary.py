#using Qwen generate summary

from datasets import load_dataset

import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import json
# Set OpenAI's API key and API base to use vLLM's API server.
from scipy.stats import kendalltau


prompt = "You are a helpful summary assistant. You can help users summarize news in two sentences."


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


#generate qwen summary
# data = json.load(open('/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_cnndm_average.json'))
data = json.load(open('./data/filter_annotations_summeval.jsonl'))

for i in tqdm(range(len(data))):
    news = data[i]['article']
    
    chat_response = client.chat.completions.create(
        model="Qwen/Qwen1.5-32B-Chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Summarize the news in two sentences.\nNews: "+news+'\nSummary: '},
            
        ],
        temperature=0,
        max_tokens=70,#40
    )
    
    res = chat_response.choices[0].message.content
    print(res)    
    data[i]['qwen_summary'] = res
    
#save to json
#./filter_annotations_summeval_llama2_summary.jsonl
#/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_cnndm_average_with_yi34b.json
# with open('./data/likert_evaluation_results_cnndm_average_with_qwen32b.jsonl', 'w') as f:
with open('./data/filter_annotations_summeval_qwen32b_summary.jsonl', 'w') as f:
    json.dump(data, f, indent=4)
