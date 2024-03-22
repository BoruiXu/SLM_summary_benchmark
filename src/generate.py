import torch
import datasets
import numpy as np
import pandas as pd
from util import eval_logger
from openai import OpenAI 
from tqdm import tqdm

#先处理api的情况
#然后处理加载模型的
#加载模型的时候，需要考虑是不是用chat template

def generate_summary(
    model,
    model_args,
    dataset,
    num_fewshot,
    batch_size,
    prompt,
    device,
    gen_args_list):
    
    if(model=="api_chat"):
        if(num_fewshot==0):
            #TODO need modify
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8222/v1"
            
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            
             
            atricle_list = dataset['article'] #TODO modify to fit the dataset
            summary_list = []
            
            loop_len = len(atricle_list)
            for i in tqdm(range(loop_len)):
                news = atricle_list[i]
                
                chat_response = client.chat.completions.create(
                    model=model_args,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Article: "+news+'\n\n  Summary: '},    
                    ],
                    temperature=0
                )
                
                res = chat_response.choices[0].message.content
                summary_list.append(res)
            return [{"article":a,"summary":s} for a, s in zip(atricle_list, summary_list)]
            
        else:
            eval_logger.error("Fewshot is not supported for api model at present")
            return
        
    elif(model=="hf"):
        print("1111")
        
        
        
    else:
        eval_logger.error("Invalid model type")
        return
    
    
