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

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = model.eval()

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False,
} 
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
    result_name = llama2_files[i].split('.')[0] + '_qwen2-0.5b-ins.jsonl'

    average_score1 = 0
    average_score2 = 0
    average_len1 = 0
    average_len2 = 0
    for j in tqdm(range(len(llama2))):
        news = llama2[j]['article']
        reference = llama2[j]['qwen_reference_summary']
        
        prompt_user1 = f'News: {news}\nSummarize the news in two sentences. Summary:'
        
        messages = [    {"role": "system", "content": prompt_sys1}, 
                        {"role": "user", "content": prompt_user1}, ]
        
        # response, history = model.chat(tokenizer, f'News: {news}\nSummarize the news in two sentences. Summary:', history=[],do_sample=False)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=500,
            do_sample= False,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response1 = clean_summary(output)
        len1 = len(response1.split(' '))
        score1 = BertScore(reference, response1)['f1'][0]*100
        print(response1)
        print("*"*30)
        prompt_user1 = f'News: {news}\nSummarize the news in two sentences. Summary:'
        messages = [    {"role": "system", "content": prompt_sys2}, 
                        {"role": "user", "content": prompt_user1}, ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=500,
            do_sample= False,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response2 = clean_summary(output)
        len2 = len(response2.split(' '))
        score2 = BertScore(reference, response2)['f1'][0]*100
        print(response2)
        
        #f1 bertscore
        # score2 = BertScore(reference, response)['f1'][0]
        average_score1 += score1
        average_score2 += score2
        average_len1 += len1
        average_len2 += len2
        
        print(f'score1: {score1}, score2: {score2}, len1: {len1}, len2: {len2}')
        print(f'average1: {average_score1/(j+1)}, average2: {average_score2/(j+1)}, average_len1: {average_len1/(j+1)}, average_len2: {average_len2/(j+1)}')
        
        tmp_dict = dict()
        tmp_dict['doc_id'] = j
        tmp_dict['article'] = news
        tmp_dict['target'] = reference
        tmp_dict['filtered_resps1'] = response1
        tmp_dict['bertscore_f1_1'] = score1
        tmp_dict['filtered_resps2'] = response2
        tmp_dict['bertscore_f1_2'] = score2
        
        
        result_list.append(tmp_dict)
         
        
    average_result[name+"_score_prompt1"] = average_score1/len(llama2)
    average_result[name+"_score_prompt2"] = average_score2/len(llama2)
    average_result[name+"_len1_prompt1"] = average_len1/len(llama2)
    average_result[name+"_len2_prompt2"] = average_len2/len(llama2)
    
    #save result to jsonl
    with open('./diff_prompt/qwen2-0.5b-ins/'+result_name, 'w') as f:
        json.dump(result_list, f, indent=4)
    with open('./diff_prompt/qwen2-0.5b-ins/average.jsonl', 'w') as f:
        json.dump([average_result], f, indent=4)
    










