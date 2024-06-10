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
import os

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8880/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

llm_model = "Qwen/Qwen2-72B-Instruct"#"Qwen/Qwen1.5-72B-Chat"#"davidkim205/Rhea-72b-v0.5"#"meta-llama/Llama-2-70b-chat-hf"#"Qwen/Qwen1.5-72B-Chat"


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
    
    if("chat" in model_name or "Chat" in model_name or "Instruct" in model_name):
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
        # print(res)
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


def loop_score_api_chat(eval_dataset, client, model_name, prompt,aspect ='relevance',temperature = 0):
    
    average_score = 0
    for i in range(len(eval_dataset)):
            news = eval_dataset[i]['doc']['article']
            summary = eval_dataset[i]['filtered_resps'][0]
            
            tmp_score = 1
            if(aspect=='faithfulness'):
                tmp_score = 0
            
            if(summary=="" or len(summary.split(' '))<5):
                print(f"empty summary, score({aspect}) is {tmp_score}")
            else:
                user_input = "News: "+news+"\nGenerated summary: "+summary
                tmp_score = (score_api_chat(client, model_name, prompt, user_input))
            #user_input = "News: "+news+"\nGenerated summary: "+summary
            #tmp_score = (score_api_chat(client, model_name, prompt, user_input))
            eval_dataset[i][str(model_name)+'_'+aspect+'_score'] = tmp_score
            average_score+=tmp_score
        
    return eval_dataset, average_score/len(eval_dataset)   



def evaluate(path, client, llm_model, few_shot = 0):
    
    aspect_list = ['relevance','coherence','faithfulness']
    average_dict = {}
    res_list = []
    
    dataset_name = path.split("/")[-1].split(".")[0]
    eval_model = path.split("/")[-2].split(".")[0]
    print(f'evaluating mdeol: {eval_model}')
    print("evaluating dataset: ", dataset_name)
    model_name = llm_model.split("/")[1]
    
    eval_dataset = json.load(open(path))
    
    for aspect in aspect_list:
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
        res_list,average_score = loop_score_api_chat(eval_dataset, client, llm_model, prompt,aspect=aspect)
        
        average_dict[aspect] = average_score
    
    
    if os.path.exists('./qwen2_72b_eva_data/'+str(eval_model)):
        pass
    else:
        os.makedirs('./qwen2_72b_eva_data/'+str(eval_model))
    
    #save the result
    if(few_shot==0):
        save_name = str(model_name)+'_'+str(dataset_name)+'_eva.jsonl'
        average_score_path = str(model_name)+'_'+str(dataset_name)+'_average.jsonl'
        
    
    with open('./qwen2_72b_eva_data/'+str(eval_model)+'/'+save_name, 'w') as fp:
        json.dump(res_list, fp, indent=4)
    with open('./qwen2_72b_eva_data/'+str(eval_model)+'/'+average_score_path, 'w') as fp:
        json.dump(average_dict, fp)

    
if __name__ == "__main__":
    
    #evaluate_list = ['microsoft_phi-2','internlm_internlm2-1_8b', 'internlm_internlm2-chat-1_8b-sft','mistralai_Mistral-7B-v0.1','mistralai_Mistral-7B-Instruct-v0.2',
    #                'TinyLlama_TinyLlama-1.1B-Chat-v1.0', 'openbmb_MiniCPM-2B-sft-bf16', 'google_gemma-1.1-2b-it', 'bigscience_bloom-560m',
    #                'Qwen_Qwen1.5-0.5B-Chat','Qwen_Qwen1.5-1.8B-Chat']
    evaluate_list = ['microsoft_phi-2','internlm_internlm2-1_8b', 'internlm_internlm2-chat-1_8b-sft','mistralai_Mistral-7B-v0.1','mistralai_Mistral-7B-Instruct-v0.2',
                    'TinyLlama_TinyLlama-1.1B-Chat-v1.0', 'openbmb_MiniCPM-2B-sft-bf16', 'google_gemma-1.1-2b-it', 'bigscience_bloom-560m',
                    'Qwen_Qwen1.5-0.5B-Chat','ahxt_LiteLlama-460M-1T','EleutherAI_gpt-neo-1.3B','microsoft_Phi-3-mini-4k-instruct']
    #evaluate_list = ['ahxt_LiteLlama-460M-1T','EleutherAI_gpt-neo-1.3B','microsoft_Phi-3-mini-4k-instruct']
    for eval_model in evaluate_list:
        base_path = '/home/xbr/LLM/lm-evaluation-harness/neurips_summary_benchmark/0shot/'+eval_model+'/'
        files = os.listdir(base_path)
        jsonl_files = [file for file in files if file.endswith(".jsonl")]
        for f in jsonl_files:
            full_path = os.path.join(base_path, f)
            evaluate(full_path, client, llm_model)
    # p = '/home/xbr/LLM/lm-evaluation-harness/neurips_summary_benchmark/0shot/microsoft_phi-2/cnndm_qwen_pretrained__microsoft__phi-2__trust_remote_code__True.jsonl' #'/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_xsum_average.json'
    # evaluate(p, client, llm_model)
