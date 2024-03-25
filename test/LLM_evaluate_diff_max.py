#this noetbook focus on the evaluation of summary quality using LLM
from datasets import load_dataset
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import json
# Set OpenAI's API key and API base to use vLLM's API server.
from scipy.stats import kendalltau, spearmanr
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8880/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

llm_model = "Qwen/Qwen1.5-72B-Chat"


#define function block

#genreate prompt
def generate_prompt(few_shot: int, aspect:str, refer_news = None, refer_summary = None, refer_score = None):
    prompt = "You are a helpful summary assistant."
    if(aspect!="Faithfulness"):
        base_prompt = "In this task, you will be provided with a news article and a generated summary.\n"\
                "Your task is to rate the " + aspect +" of the generated summary with a score from 1 to 5, "\
                "where 1 is the lowest and 5 is the highest.\n"\
                "Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n"
        if(few_shot==0):
            prompt = base_prompt+ "Example Response:\n"+aspect+" (1-5): 3"
        else:
            shot_num = len(refer_news)
            if shot_num>2:
                shot_num = 2
            if(shot_num==1):
                base_prompt+="This is an example:\n"
            else:
                base_prompt+="These are examples:\n"
            
            for i in range(shot_num):
                base_prompt = base_prompt + "News: "+refer_news[i]+"\nGenerated summary: "+refer_summary[i]+"\n"+aspect+" (1-5): "+str(refer_score[i])+"\n"
            prompt = base_prompt
    else:
        base_prompt = "In this task, you will be provided with a news article and a generated summary.\n"\
                "Your task is to rate the " + aspect +" of the generated summary with a score 0 or 1, "\
                "where 1 is not fact and 1 is fact.\n"\
                "Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n"
        
        if(few_shot==0):
            prompt = base_prompt+ "Example response:\n"+aspect+" (0,1): 1"
        else:
            shot_num = len(refer_news)

            if shot_num>2:
                shot_num = 2
            if(shot_num==1):
                base_prompt+="This is an example:\n"
            else:
                base_prompt+="These are examples:\n"
            for i in range(shot_num):
                base_prompt = base_prompt + "News: "+refer_news[i]+"\nGenerated summary: "+refer_summary[i]+"\n"+aspect+" (0,1): "+str(refer_score[i])+"\n"
            prompt = base_prompt
                   
    return prompt
    

#get socre using openai api
def score_api_chat(client, model_name, prompt, user_input, temperature = 0):
    chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
                
            ],
            temperature=temperature
        )
        # "\n\nReference summary: "
        #         +reference_summary+
    res = chat_response.choices[0].message.content
        
    match = re.search(r':\s*(\d+)', res)   
    score = match.group(1) if match is not None else 1
    return int(score)


def loop_score_api_chat(news_list, summary_list, client, model_name, aspect, 
                        refer_news = None, refer_summary = None, refer_score = None,temperature = 0):
    
    score_list = []
    prompt = ""
    if(refer_news is None):
        prompt = generate_prompt(0, aspect)  
    else:
        prompt = generate_prompt(1, aspect, refer_news, refer_summary, refer_score)

    
    for i in range(len(news_list)):
            news = news_list[i]
            summary = summary_list[i]
            user_input = "News: "+news+"\nGenerated summary: "+summary
            score_list.append(score_api_chat(client, model_name, prompt, user_input))
    return score_list   

def correlation_score(dict1, dict2):
    #system level
    tmp_list1 = []
    tmp_list2 = []
    for i in dict1.keys():
        tmp_list1.append(np.mean(dict1[i]))
        tmp_list2.append(np.mean(dict2[i]))
    print("kendalltau correlation of system level is ", kendalltau(tmp_list1, tmp_list2)[0])
    print("spearmans correlation of system level is ", spearmanr(tmp_list1, tmp_list2)[0])
    
    #summary level
    total_corr = 0
    total_corr2 = 0
    
    for i in dict1.keys():
        total_corr+=kendalltau(dict1[i], dict2[i])[0]
        total_corr2+=spearmanr(dict1[i], dict2[i])[0]
    print("kendalltau correlation of summary level is ", total_corr/len(dict1.keys()))
    print("spearmans correlation of summary level is ", total_corr2/len(dict1.keys()))
    

#want to evaluate without example first
#then find which is the best and worst, then use them as template

def get_template_list(candiate_news, candiate_summary, candiate_score,client, llm_model, aspect):
    res_news_list = []
    res_summary_list = []
    res_score_list = []
    
    print("evaluating without example first")
    #first evaluate
    score_list = loop_score_api_chat(candiate_news, candiate_summary, client, llm_model, aspect, 
                                     refer_news = None, refer_summary = None, refer_score = None)
    
    for i in range(len(score_list)):
        score_list[i] = abs(score_list[i]-candiate_score[i])
    
    
    max_diff_score = max(score_list)
    
    max_indexes = [index for index, value in enumerate(score_list) if value == max_diff_score]
    
    print(max_indexes)
    max_index = 0 #max_indexes[0]
    
    print(max_index)
    
    res_news_list.append(candiate_news[max_index])
    res_summary_list.append(candiate_news[max_index])
    res_score_list.append(candiate_score[max_index])
    
    
    return res_news_list, res_summary_list, res_score_list, max_index



cnndm_liker = pd.read_json('/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_cnndm_average.json')

target_dataset = cnndm_liker
template_dataset = cnndm_liker

aspect = "relevance"


model_list = list(set(target_dataset['model'].tolist()))
model_list.remove('reference')


candiate_news = template_dataset[template_dataset["model"]=="reference"]["article"].to_list()
candiate_summary = template_dataset[template_dataset["model"]=="reference"]["summary"].to_list()
candiate_score = template_dataset[template_dataset["model"]=="reference"][aspect].to_list()

#template_news, template_summary, template_score, max_index =  get_template_list(candiate_news, candiate_summary, candiate_score, client, llm_model, aspect)
template_news = None
template_summary = None
template_score = None 
max_index = 999


model_eva_dict= {}
human_eva_dict = {}

for m in model_list:
    print("evaluating model: ", m)
    tmp_dataset = target_dataset[(target_dataset['model']==m )]
    
    tmp_news_list = tmp_dataset['article'].tolist()
    
    tmp_summary_list = tmp_dataset['summary'].tolist()
    tmp_score_list = tmp_dataset[aspect].tolist()
    score_list = loop_score_api_chat(tmp_news_list, tmp_summary_list, client, llm_model, aspect, 
                                     refer_news = template_news, refer_summary = template_summary, refer_score = template_score)
    
    model_eva_dict[m] = score_list
    human_eva_dict[m] = tmp_score_list
    

    
#save the result
with open('./evaluate_result/'+'qwen72_cnndm_'+aspect+'_template_cnndm_diff_max'+str(max_index)+'_eva.json', 'w') as fp:
    json.dump(model_eva_dict, fp)

correlation_score(model_eva_dict, human_eva_dict)
