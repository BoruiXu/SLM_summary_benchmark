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

llm_model = "meta-llama/Meta-Llama-3-70B-Instruct"#"Qwen/Qwen1.5-32B-Chat"#"davidkim205/Rhea-72b-v0.5"#"meta-llama/Llama-2-70b-chat-hf"#"Qwen/Qwen1.5-72B-Chat"


#define function block

#genreate prompt
def generate_prompt(few_shot: int, aspect:str, refer_news = None, refer_summary = None, refer_score = None):
    prompt = "You are a helpful summary assistant."
    aspect = aspect.split("_")[-1]
    
    if(aspect!="faithfulness"):
        base_prompt = "In this task, you will be provided with a news article and a generated summary.\n"\
                "Your task is to rate the " + aspect +" of the generated summary with a score from 1 to 5, "\
                "where 1 is the lowest and 5 is the highest.\n"\
                "Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n"
        if(few_shot==0):
            prompt = base_prompt+ "Example response:\n"+aspect+" (1-5): 2"
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
    
    if("chat" in model_name or "Chat" in model_name or "meta-llama/Meta-Llama-3-70B-Instruct" == model_name):
        chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input},
                    
                ],
                temperature=temperature,
                max_tokens=15,
                
            )
            # "\n\nReference summary: "
            #         +reference_summary+
        res = chat_response.choices[0].message.content
        #print(res)
        if('faithfulness' in prompt):    
            match = re.search(r':\s*(\d+)', res)
            
            if(match is not None):
                score = match.group(1)
            else:
                match = re.search(r'^\d$', res)
                score = match.group() if match is not None else 0
                
            score = int(score)
            if(score>1):
                score = 1
            if(score<0):
                score = 0
            print(f"Res: {res}, Score: {score}")
            return int(score)
        else:
            match = re.search(r':\s*(\d+)', res)
            if(match is not None):
                score = match.group(1)
            else:
                match = re.search(r'^\d$', res)
                score = match.group() if match is not None else 0   
            # score = match.group(1) if match is not None else 1
            score = int(score)
            if(score>5):
                score = 5
            if(score<1):
                score = 1
            print(f"Res: {res}, Score: {score}")
            return int(score)
    else:
        completion = client.completions.create(model="abacusai/Smaug-72B-v0.1",
                                      prompt=prompt+"\n"+user_input,
                                      temperature=0)
        print("Result:", completion.choices[0].text)


def loop_score_api_chat(news_list, summary_list, client, model_name, prompt,temperature = 0):
    
    score_list = []
    
    
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
    
# max_index = 0
# min_index = 0    
# def get_template_list(news_list,summary_list,score_list):
    
#     res_news_list = []
#     res_summary_list = []
#     res_score_list = []
    
#     max_score = max(score_list)
#     min_score = min(score_list)
#     # max_index = score_list.index(max_score)
#     # min_index = score_list.index(min_score)
    
#     max_indexes = [index for index, value in enumerate(score_list) if value == max_score]
#     min_indexes = [index for index, value in enumerate(score_list) if value == min_score]
    
#     print(max_indexes, min_indexes)
#     max_index = max_indexes[1]
#     min_index = min_indexes[0]
    
#     print(max_index, min_index)
    
#     res_news_list.append(news_list[max_index])
#     res_summary_list.append(summary_list[max_index])
#     res_score_list.append(score_list[max_index])
    
#     res_news_list.append(news_list[min_index])
#     res_summary_list.append(summary_list[min_index])
#     res_score_list.append(score_list[min_index])
    
#     return res_news_list, res_summary_list, res_score_list, max_index, min_index

def evaluate(path, aspect, client, llm_model, reference_model = 'reference', few_shot = 0):
    
    dataset_name = path.split("/")[-1].split(".")[0]
    print("evaluating dataset: ", dataset_name)
    model_name = llm_model.split("/")[1]
    
    target_dataset = pd.read_json(path)
    model_list = list(set(target_dataset['model'].tolist()))
    model_list.remove(reference_model)
    model_list = sorted(model_list)
    
    
    candiate_news = target_dataset[target_dataset["model"]==reference_model]["article"].to_list()
    candiate_summary = target_dataset[target_dataset["model"]==reference_model]["summary"].to_list()
    candiate_score = target_dataset[target_dataset["model"]==reference_model][aspect].to_list()
    
    prompt = ""
    
    if(few_shot!=0):
        # template_news, template_summary, template_score, max_index, min_index =  get_template_list(candiate_news, candiate_summary, candiate_score)
        # prompt = generate_prompt(1, aspect, refer_news, refer_summary, refer_score)
        print("Not implement yet")
    else:
        prompt = generate_prompt(0, aspect)
    
    print("Prompt:")
    print(prompt)
    #save result    
    model_eva_dict= {}
    human_eva_dict = {}

    for m in model_list:
        print("evaluating model: ", m)
        tmp_dataset = target_dataset[(target_dataset['model']==m )]
        
        tmp_news_list = tmp_dataset['article'].tolist()
        
        tmp_summary_list = tmp_dataset['summary'].tolist()
        tmp_score_list = tmp_dataset[aspect].tolist()
        score_list = loop_score_api_chat(tmp_news_list, tmp_summary_list, client, llm_model, prompt)
        
        model_eva_dict[m] = score_list
        human_eva_dict[m] = tmp_score_list  
    
    #save the result
    if(few_shot==0):
        save_name = str(model_name)+'_'+str(dataset_name)+'_'+str(aspect)+'_eva.json'
        human_save = 'human_score_'+str(dataset_name)+'_'+str(aspect)+'_eva.json'
        prompt_save = 'prompt_'+str(dataset_name)+'_'+str(aspect)+'_eva.json'
    
    with open('./LLM_evaluation_correlation_with_human/'+save_name, 'w') as fp:
        json.dump(model_eva_dict, fp)
    with open('./LLM_evaluation_correlation_with_human/'+human_save, 'w') as fp:
        json.dump(human_eva_dict, fp)
    with open('./LLM_evaluation_correlation_with_human/'+prompt_save, 'w') as fp:
        json.dump(prompt, fp)

    correlation_score(model_eva_dict, human_eva_dict)
    
if __name__ == "__main__":
    
    p = './data/likert_evaluation_results_cnndm_average.json'
    #p = './data/filter_annotations_summeval.jsonl' #'/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_xsum_average.json'
    aspect = "faithfulness"
    evaluate(p, aspect, client, llm_model, reference_model = 'reference')
