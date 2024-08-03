import json
import pandas as pd
from evaluate import load
bert_score = load("bertscore")
def BertScore(refs, preds):
    bert_score_res = bert_score.compute(predictions=[preds], references=[refs], model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    return bert_score_res


#evaluate ChatGLM3
import re
from tqdm import tqdm
def clean_summary(completion):
        completion = completion[2:] if completion.startswith("\n\n") else completion
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
        return completion
    
    
qwen1_5_files = ['xsum_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl',
                'newsroom_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl',
                'cnndm_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl',
                'bbc2024_sample_500_0k5_1k5_qwen1.5_72b_summary_no_len_limit.jsonl']
base_qwen1_5_path = '/home/xbr/LLM/summary_benchmark/dataset/sample_500_qwen72b_summary'


llama2_files = ['xsum_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl',
                'newsroom_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl',
                'cnndm_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl',
                'bbc2024_sample_500_0k5_1k5_llama2_70b_summary_no_len_limit.jsonl']
base_llama2_path = '/home/xbr/LLM/summary_benchmark/dataset/sample_500_llama2_70b_summary'
 

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-1_8b", trust_remote_code=True)
model = AutoModel.from_pretrained("internlm/internlm2-chat-1_8b", trust_remote_code=True).half().cuda()
model = model.eval()

average_result = dict()
for i in range(len(llama2_files)):
    name = llama2_files[i].split('_')[0]
    if("bbc2024" not in name):
        continue
    print(name)
    
    f1 = base_llama2_path + '/' + llama2_files[i]
    #use json to load f1 
    llama2 = json.load(open(f1))
    
    
    result_list = list()
    result_name = llama2_files[i] + '_internlm_chat_1.8B.jsonl'

    average_score = 0
    average_len1 = 0
    average_len2 = 0
    average_len3 = 0
    average_len4 = 0
    
    for j in tqdm(range(len(llama2))):
        news = llama2[j]['article']
        reference = llama2[j]['qwen_reference_summary']
        
        response1, history = model.chat(tokenizer, f'News: {news}\nSummarize the news in 50 words. Summary:', history=[],do_sample=False)
        response1 = clean_summary(response1)
        
        response2, history = model.chat(tokenizer, f'News: {news}\nSummarize the news in 60 words. Summary:', history=[],do_sample=False)
        response2 = clean_summary(response2)
        
        response3, history = model.chat(tokenizer, f'News: {news}\nSummarize the news in 70 words. Summary:', history=[],do_sample=False)
        response3 = clean_summary(response3)
        
        response4, history = model.chat(tokenizer, f'News: {news}\nSummarize the news in 80 words. Summary:', history=[],do_sample=False)
        response4 = clean_summary(response4)
        
        #f1 bertscore
        # score = BertScore(reference, response)['f1'][0]
        # average_score += score
        average_len1 += len(response1.split(' '))
        average_len2 += len(response2.split(' '))
        average_len3 += len(response3.split(' '))
        average_len4 += len(response4.split(' '))
        
        
        tmp_dict = dict()
        tmp_dict['article'] = news
        tmp_dict['target'] = reference
        tmp_dict['filtered_resps_50word'] = response1
        tmp_dict['filtered_resps_60word'] = response2
        tmp_dict['filtered_resps_70word'] = response3
        tmp_dict['filtered_resps_80word'] = response4
        # tmp_dict['bertscore_f1'] = score
        
        result_list.append(tmp_dict)
         
        
    average_result[name+"_len_1_sec"] = average_len1/len(llama2)
    average_result[name+"_len_2_sec"] = average_len2/len(llama2)
    average_result[name+"_len_3_sec"] = average_len3/len(llama2)
    average_result[name+"_len_4_sec"] = average_len4/len(llama2)
    
    #save result to jsonl
    with open('./diff_prompt/internLM2_chat_diff_word/'+result_name, 'w') as f:
        json.dump(result_list, f, indent=4)
    with open('./diff_prompt/internLM2_chat_diff_word/average.jsonl', 'w') as f:
        json.dump([average_result], f, indent=4)
    
