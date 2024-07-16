import json
import pandas as pd
from evaluate import load
from transformers import AutoTokenizer, AutoModel,pipeline,AutoModelForCausalLM
import transformers


bert_score = load("bertscore")
def BertScore(refs, preds):
    bert_score_res = bert_score.compute(predictions=[preds], references=[refs], model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    return bert_score_res


#evaluate ChatGLM3
import re
from tqdm import tqdm

from openai import OpenAI          
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8880/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def clean_summary(completion):
        completion = completion[2:] if completion.startswith("\n\n") else completion
        completion = completion.replace('Summarize the news in two sentences. Summary: ','')
        completion = completion.replace('The news summary is: \"', '')
        completion_list = completion.split('\n')
        if(len(completion_list)<3):
            completion = completion_list[-1]
        else:
            completion = completion.split('\n')[2]
        candidate = completion
        complete_sentences = re.findall(r'[^.!?]*[.!?]', completion)
        # 将匹配到的完整句子连接成一个新的字符串
        completion = ''.join(complete_sentences)
        if(completion==''):
            completion = candidate
        completion = completion.strip()
        return completion
    
    
# qwen1_5_files = ['xsum_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl',
#                 'newsroom_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl',
#                 'cnndm_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl',
#                 'bbc2024_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl']
# base_qwen1_5_path = '/home/xbr/LLM/summary_benchmark/dataset/sample_500_qwen72b_summary'


llama2_files = ['xsum_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl',
                'newsroom_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl',
                'cnndm_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl',
                'bbc2024_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl']
base_llama2_path = '/home/xbr/LLM/summary_benchmark/dataset/sample_500_llama2_70b_summary'
 

from transformers import AutoTokenizer, AutoModel,pipeline,AutoModelForCausalLM
device = "cuda" # the device to load the model onto


prompt_sys1= "You are a helpful summary assistant. You can help users summarize news in two sentences."

prompt_sys2= "You are a helpful summary assistant. You can summarize the following news article in two sentences, highlighting the key points and main events."


average_result = dict()
for i in range(len(llama2_files)):
    name = llama2_files[i].split('_')[0]
    if(name!="bbc2024"):
        continue
    print(name)
    f1 = base_llama2_path + '/' + llama2_files[i]
    #use json to load f1 
    llama2 = json.load(open(f1))
    
    
    result_list = list()
    result_name = llama2_files[i].split('.')[0] + '_llama3-70b-ins.jsonl'

    average_score1 = 0
    average_score2 = 0
    average_score3 = 0
    
    average_len1 = 0
    average_len2 = 0
    average_len3 = 0
    
    for j in tqdm(range(len(llama2))):
        news = llama2[j]['article']
        reference = llama2[j]['qwen_reference_summary']
        
        prompt_user1 = f'News: {news}\nSummarize the news in two sentences. Summary:'
        
        # messages = prompt_sys1+'\n'+prompt_user1
        
        chat_response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=[
                {"role": "user", "content": prompt_user1},
            ],
            temperature=0,
          
        )
        
        res = chat_response.choices[0].message.content
        
        response1 = clean_summary(res)
        len1 = len(response1.split(' '))
        score1 = BertScore(reference, response1)['f1'][0]*100
        print(response1)
        print("*"*30)
        
        
        prompt_user1 = f'News: {news}\nSummarize the news in two sentences. Summary:'
        
        chat_response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=[
                {"role": "system", "content": prompt_sys1},
                {"role": "user", "content": prompt_user1},
            ],
            temperature=0,

        )
        
        res = chat_response.choices[0].message.content
        response2 = clean_summary(res)
        len2 = len(response2.split(' '))
        score2 = BertScore(reference, response2)['f1'][0]*100
        print(response2)
        print("*"*30)
        
        chat_response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            messages=[
                {"role": "system", "content": prompt_sys2},
                {"role": "user", "content": prompt_user1},
            ],
            temperature=0,

        )
        
        res = chat_response.choices[0].message.content
        response3 = clean_summary(res)
        len3 = len(response3.split(' '))
        score3 = BertScore(reference, response3)['f1'][0]*100
        print(response3)
        print("*"*30)
        
        
        
        #f1 bertscore
        # score2 = BertScore(reference, response)['f1'][0]
        average_score1 += score1
        average_score2 += score2
        average_score3 += score3
        
        average_len1 += len1
        average_len2 += len2
        average_len3 += len3
        
        print(f'score1: {score1}, score2: {score2}, score3: {score3}, len1: {len1}, len2: {len2}, len3: {len3}')
        print(f'average1: {average_score1/(j+1)}, average2: {average_score2/(j+1)}, average3: {average_score3/(j+1)}') 
        print(f'average_len1: {average_len1/(j+1)}, average_len2: {average_len2/(j+1)}, average_len3: {average_len3/(j+1)}')
        
        tmp_dict = dict()
        tmp_dict['doc_id'] = j
        tmp_dict['article'] = news
        tmp_dict['target'] = reference
        tmp_dict['prompt1'] = ''
        tmp_dict['filtered_resps1'] = response1
        tmp_dict['bertscore_f1_1'] = score1
        tmp_dict['prompt2'] = prompt_sys1
        tmp_dict['filtered_resps2'] = response2
        tmp_dict['bertscore_f1_2'] = score2
        tmp_dict['prompt3'] = prompt_sys2
        tmp_dict['filtered_resps3'] = response3
        tmp_dict['bertscore_f1_3'] = score3
        
        result_list.append(tmp_dict)
         
        
    average_result[name+"_score_prompt1"] = average_score1/len(llama2)
    average_result[name+"_score_prompt2"] = average_score2/len(llama2)
    average_result[name+"_score_prompt3"] = average_score3/len(llama2)
    average_result[name+"_len1_prompt1"] = average_len1/len(llama2)
    average_result[name+"_len2_prompt2"] = average_len2/len(llama2)
    average_result[name+"_len3_prompt3"] = average_len3/len(llama2)
    
    #save result to jsonl
    with open('./diff_prompt/llama3-70b-ins/'+result_name, 'w') as f:
        json.dump(result_list, f, indent=4)
    with open('./diff_prompt/llama3-70b-ins/average.jsonl', 'w') as f:
        json.dump([average_result], f, indent=4)
    










