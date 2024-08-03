import json
import pandas as pd
from evaluate import load
import os
import torch
bert_score = load("bertscore")
def BertScore(refs, preds):
    bert_score_res = bert_score.compute(predictions=[preds], references=[refs], model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    return bert_score_res

from transformers import AutoTokenizer, AutoModelForSequenceClassification
factkb_tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
factkb_model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels=2, device_map="auto")

def FKB(response1, news):
    input_factkb = [[response1, news]]
    factkb_tokens = factkb_tokenizer(input_factkb, return_tensors="pt", padding="max_length", truncation=True).to(factkb_model.device)
    factkb_logits = factkb_model(**factkb_tokens).logits
    factkb_res = torch.softmax(factkb_logits, dim=1)
    
    return float(factkb_res[0][1])*100

#evaluate ChatGLM3
import re
from tqdm import tqdm

# model_name = 'Qwen/Qwen2-1.5B-Instruct'
model_list = ["Qwen/Qwen2-0.5B-Instruct"]
# print(model_name)
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
 
for model_name in model_list:
    print(model_name)
    
    from transformers import AutoTokenizer, AutoModel,pipeline,AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained( model_name, device_map="cuda", torch_dtype="auto",trust_remote_code=True)
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
    #1 ,2,3,4 
    #50,60,70,80
    # prompt_sys1= "You are a helpful summary assistant. You can help users summarize news in 50 words."

    # prompt_sys2= "You are a helpful summary assistant. You can summarize the following news article in two sentences, highlighting the key points and main events."

    # prompt_sys2= "You are a helpful summary assistant. You can help users summarize news in 60 words."

    # prompt_sys3= "You are a helpful summary assistant. You can help users summarize news in 70 words."

    # prompt_sys4= "You are a helpful summary assistant. You can help users summarize news in 80 words."

    average_result = dict()
    for i in range(len(llama2_files)):
        name = llama2_files[i].split('_')[0]
        # if(name!="bbc2024"):
        #     continue
        print(name)
        f1 = base_llama2_path + '/' + llama2_files[i]
        #use json to load f1 
        llama2 = json.load(open(f1))
        
        
        result_list = list()
        result_name = llama2_files[i].split('.')[0] + '_'+model_name.split('/')[1]+'.jsonl'

        average_score1 = 0
        average_score2 = 0
        average_score3 = 0
        average_score4 = 0
        average_len1 = 0
        average_len2 = 0
        average_len3 = 0
        average_len4 = 0
        
        average_fkb2 = 0
        for j in tqdm(range(len(llama2))):
            news = llama2[j]['article']
            reference = llama2[j]['qwen_reference_summary']
            
            # prompt_user1 = f'News: {news}\nSummarize the news in one sentence. Summary:'
            # prompt_user1 = f'News: {news}\nSummarize the news in 50 words. Summary:'
            # prompt_user1 = f'News: {news}\nSummarize the news in fifty words. Summary:'
            
            # # messages = [    {"role": "system", "content": prompt_sys1}, 
            # #                 {"role": "user", "content": prompt_user1}, ]
            # messages1 = [   {"role": "user", "content": prompt_user1}]
            
            # # response, history = model.chat(tokenizer, f'News: {news}\nSummarize the news in two sentences. Summary:', history=[],do_sample=False)
            # output = pipe(messages1, **generation_args)[0]['generated_text'] 
            # response1 = clean_summary(output)
            # len1 = len(response1.split(' '))
            # score1 = BertScore(reference, response1)['f1'][0]*100
            # print(response1)
            # print("*"*30)
            prompt_user2 = f'News: {news}\nSummarize the news in two sentences. Summary:'
            # prompt_user2 = f'News: {news}\nSummarize the news in 60 words. Summary:'
            # prompt_user2 = f'News: {news}\nSummarize the news in sixty words. Summary:'
            # messages = [    {"role": "system", "content": prompt_sys2}, 
            #                 {"role": "user", "content": prompt_user2}, ]
            messages2 = [   {"role": "user", "content": prompt_user2}]
            output = pipe(messages2, **generation_args)[0]['generated_text']
            response2 = clean_summary(output)
            len2 = len(response2.split(' '))
            score2 = BertScore(reference, response2)['f1'][0]*100
            fkb2 = FKB(response2, news)
            print(response2)
            
            
            print("*"*30)
            # prompt_user3 = f'News: {news}\nSummarize the news in three sentences. Summary:'
            # prompt_user3 = f'News: {news}\nSummarize the news in 70 words. Summary:'
            # prompt_user3 = f'News: {news}\nSummarize the news in seventy words. Summary:'
            # # messages = [    {"role": "system", "content": prompt_sys3}, 
            # #                 {"role": "user", "content": prompt_user3}, ]
            # messages3 = [   {"role": "user", "content": prompt_user3}]
            # output = pipe(messages3, **generation_args)[0]['generated_text']
            # response3 = clean_summary(output)
            # len3 = len(response3.split(' '))
            # score3 = BertScore(reference, response3)['f1'][0]*100
            # print(response3)
            
            
            # print("*"*30)
            # # prompt_user4 = f'News: {news}\nSummarize the news in four sentences. Summary:'
            # # prompt_user4 = f'News: {news}\nSummarize the news in 80 words. Summary:'
            # prompt_user4 = f'News: {news}\nSummarize the news in eighty words. Summary:'
            # # messages = [    {"role": "system", "content": prompt_sys4}, 
            # #                 {"role": "user", "content": prompt_user4}, ]
            # messages4 = [   {"role": "user", "content": prompt_user4}]
            # output = pipe(messages4, **generation_args)[0]['generated_text']
            # response4 = clean_summary(output)
            # len4 = len(response4.split(' '))
            # score4 = BertScore(reference, response4)['f1'][0]*100
            # print(response4)
            
            #f1 bertscore
            # score2 = BertScore(reference, response)['f1'][0]
            # average_score1 += score1
            average_score2 += score2
            # average_score3 += score3
            # average_score4 += score4
            # average_len1 += len1
            average_len2 += len2
            # average_len3 += len3
            # average_len4 += len4
            
            average_fkb2 += fkb2
            
            # print(f'score1: {score1}, score2: {score2}, len1: {len1}, len2: {len2}')
            # print(f'average1: {average_score1/(j+1)}, average2: {average_score2/(j+1)}, average3: {average_score3/(j+1)}, average4: {average_score4/(j+1)}')
            # print(f'average_len1: {average_len1/(j+1)}, average_len2: {average_len2/(j+1)}, average_len3: {average_len3/(j+1)}, average_len4: {average_len4/(j+1)}')
            print(f'average_fkb2: {average_fkb2/(j+1)}')
            tmp_dict = dict()
            tmp_dict['doc_id'] = j
            tmp_dict['article'] = news
            tmp_dict['target'] = reference
            # tmp_dict['prompt1'] = messages1
            # tmp_dict['filtered_resps1'] = response1
            # tmp_dict['bertscore_f1_1'] = score1
            tmp_dict['prompt2'] = messages2
            tmp_dict['filtered_resps2'] = response2
            tmp_dict['bertscore_f1_2'] = score2
            tmp_dict['factKB2'] = fkb2
            # tmp_dict['prompt3'] = messages3
            # tmp_dict['filtered_resps3'] = response3
            # tmp_dict['bertscore_f1_3'] = score3
            # tmp_dict['prompt4'] = messages4
            # tmp_dict['filtered_resps4'] = response4
            # tmp_dict['bertscore_f1_4'] = score4
            
            
            result_list.append(tmp_dict)
            
            
        # average_result[name+"_score_prompt1"] = average_score1/len(llama2)
        average_result[name+"_score_prompt2"] = average_score2/len(llama2)
        # average_result[name+"_score_prompt3"] = average_score3/len(llama2)
        # average_result[name+"_score_prompt4"] = average_score4/len(llama2)
        
        # average_result[name+"_len1_prompt1"] = average_len1/len(llama2)
        average_result[name+"_len2_prompt2"] = average_len2/len(llama2)
        # average_result[name+"_len3_prompt3"] = average_len3/len(llama2)
        # average_result[name+"_len4_prompt4"] = average_len4/len(llama2)
        
        average_result[name+"_fkb2_prompt2"] = average_fkb2/len(llama2)
        #save result to jsonl
        
        # if not os.path.exists('./diff_prompt/'+model_name.split('/')[1]):
        #     os.makedirs('./diff_prompt/'+model_name.split('/')[1])
        
        # with open('./diff_prompt/'+model_name.split('/')[1]+'/diff_fifty_words_no_system_prompt_'+result_name, 'w') as f:
        #     json.dump(result_list, f, indent=4)
        # with open('./diff_prompt/'+model_name.split('/')[1]+'/average_fifty_words_sentence_no_system_prompt.jsonl', 'w') as f:
        #     json.dump([average_result], f, indent=4)
        
        if not os.path.exists('./factKB/'+model_name.split('/')[1]):
            os.makedirs('./factKB/'+model_name.split('/')[1])
        
        with open('./factKB/'+model_name.split('/')[1]+'/factkb_2_sentences_no_system_prompt_'+result_name, 'w') as f:
            json.dump(result_list, f, indent=4)
        with open('./factKB/'+model_name.split('/')[1]+'/average_llama2_70B_factkb_2_sentences_sentence_no_system_prompt.jsonl', 'w') as f:
            json.dump([average_result], f, indent=4)
    
