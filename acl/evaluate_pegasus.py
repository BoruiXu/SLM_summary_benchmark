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
 

from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM
model_name = 'google/pegasus-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.eval()

average_result = dict()
for i in range(len(qwen1_5_files)):
    name = qwen1_5_files[i].split('_')[0]
    print(name)
    f1 = base_qwen1_5_path + '/' + qwen1_5_files[i]
    #use json to load f1 
    llama2 = json.load(open(f1))
    
    
    result_list = list()
    result_name = qwen1_5_files[i] + '_pegasus_large.jsonl'

    average_score = 0
    for j in tqdm(range(len(llama2))):
        news = llama2[j]['article']
        reference = llama2[j]['qwen_reference_summary']
        
        batch = tokenizer(news, truncation=True, padding="longest", return_tensors="pt")
        response = model.generate(**batch)
        response = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        response = clean_summary(response)
        print(response)
        #f1 bertscore
        score = BertScore(reference, response)['f1'][0]
        average_score += score
        
        print(average_score/(j+1))
        
        tmp_dict = dict()
        tmp_dict['article'] = news
        tmp_dict['target'] = reference
        tmp_dict['filtered_resps'] = response
        tmp_dict['bertscore_f1'] = score
        
        result_list.append(tmp_dict)
         
        
    average_result[name] = average_score/len(llama2)
    
    #save result to jsonl
    with open('./qwen1_5/pegasus_large/'+result_name, 'w') as f:
        json.dump(result_list, f, indent=4)
    with open('./qwen1_5/pegasus_large/average.jsonl', 'w') as f:
        json.dump([average_result], f, indent=4)
    
