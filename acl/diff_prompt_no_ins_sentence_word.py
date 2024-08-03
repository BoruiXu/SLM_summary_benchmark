import json
import pandas as pd
from evaluate import load
from transformers import AutoTokenizer, AutoModel,pipeline,AutoModelForCausalLM
import transformers
import os

bert_score = load("bertscore")
def BertScore(refs, preds):
    bert_score_res = bert_score.compute(predictions=[preds], references=[refs], model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    return bert_score_res


#evaluate ChatGLM3
import re
from tqdm import tqdm


model_name = 'microsoft/phi-2'#'''Qwen/Qwen2-1.5B'

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: list[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
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

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model.eval()

# pipe = pipeline( 
#     "text-generation", 
#     model=model, 
#     tokenizer=tokenizer, 
# ) 

# generation_args = { 
#     "max_new_tokens": 500, 
#     "return_full_text": False, 
#     "temperature": 0.0, 
#     "do_sample": False,
# } 


def run(messages,model,tokenizer):
    # print(messages)
    
    
    model_inputs = tokenizer(messages, return_tensors="pt").to(device)
        
    stopping_criteria = stop_sequences_criteria(tokenizer, ['\n', '</s>'], model_inputs['input_ids'].shape[1], 1)

    generated_ids = model.generate(**model_inputs, max_new_tokens=250, do_sample=False)
    
    #qwen2 series nedd filter prefix    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = clean_summary(output)
    
    return response
    

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
    result_name = llama2_files[i].split('.')[0] +'_'+ model_name.split('/')[1]+'.jsonl'

    average_score1 = 0
    average_score2 = 0
    average_score3 = 0
    average_score4 = 0
    average_len1 = 0
    average_len2 = 0
    average_len3 = 0
    average_len4 = 0
    for j in tqdm(range(len(llama2))):
        news = llama2[j]['article']
        reference = llama2[j]['qwen_reference_summary']
        
        # prompt_user1 = f'News: {news}\nSummarize the news in 50 words. Summary:'
        prompt_user1 = f'News: {news}\nSummarize the news in one sentence. Summary:'
        response1 = run(prompt_user1, model, tokenizer)

        len1 = len(response1.split(' '))
        score1 = BertScore(reference, response1)['f1'][0]*100
        print(response1)
        print("*"*30)
        
        
        # prompt_user2 = f'News: {news}\nSummarize the news in 60 words. Summary:'
        prompt_user2 = f'News: {news}\nSummarize the news in two sentences. Summary:'

        response2 = run(prompt_user2, model, tokenizer)
        len2 = len(response2.split(' '))
        score2 = BertScore(reference, response2)['f1'][0]*100
        print(response2)
        print("*"*30)
        
        
        # prompt_user3 = f'News: {news}\nSummarize the news in 70 words. Summary:'
        prompt_user3 = f'News: {news}\nSummarize the news in three sentences. Summary:'

        response3 = run(prompt_user3, model, tokenizer)
        len3 = len(response3.split(' '))
        score3 = BertScore(reference, response3)['f1'][0]*100
        print(response3)
        print("*"*30)
        
        
        # prompt_user4 = f'News: {news}\nSummarize the news in 80 words. Summary:'
        prompt_user4 = f'News: {news}\nSummarize the news in four sentences. Summary:'

        response4 = run(prompt_user4, model, tokenizer)
        len4 = len(response4.split(' '))
        score4 = BertScore(reference, response4)['f1'][0]*100
        print(response4)
        print("*"*30)
        
        #f1 bertscore
        # score2 = BertScore(reference, response)['f1'][0]
        average_score1 += score1
        average_score2 += score2
        average_score3 += score3
        average_score4 += score4
        average_len1 += len1
        average_len2 += len2
        average_len3 += len3
        average_len4 += len4
        
        # print(f'score1: {score1}, score2: {score2}, len1: {len1}, len2: {len2}')
        print(f'average1: {average_score1/(j+1)}, average2: {average_score2/(j+1)}, average3: {average_score3/(j+1)}, average4: {average_score4/(j+1)}') 
        print(f'average_len1: {average_len1/(j+1)}, average_len2: {average_len2/(j+1)}, average_len3: {average_len3/(j+1)}, average_len4: {average_len4/(j+1)}')
        
        tmp_dict = dict()
        tmp_dict['doc_id'] = j
        tmp_dict['article'] = news
        tmp_dict['target'] = reference
        tmp_dict['user_prompt1'] = prompt_user1
        tmp_dict['filtered_resps1'] = response1
        tmp_dict['bertscore_f1_1'] = score1
        tmp_dict['len1'] = len1
        tmp_dict['user_prompt2'] = prompt_user2
        tmp_dict['filtered_resps2'] = response2
        tmp_dict['bertscore_f1_2'] = score2
        tmp_dict['len2'] = len2
        tmp_dict['user_prompt3'] = prompt_user3
        tmp_dict['filtered_resps3'] = response3
        tmp_dict['bertscore_f1_3'] = score3
        tmp_dict['len3'] = len3
        tmp_dict['user_prompt4'] = prompt_user4
        tmp_dict['filtered_resps4'] = response4
        tmp_dict['bertscore_f1_4'] = score4
        tmp_dict['len4'] = len4
        
        
        result_list.append(tmp_dict)
         
        
    average_result[name+"_score_prompt1"] = average_score1/len(llama2)
    average_result[name+"_score_prompt2"] = average_score2/len(llama2)
    average_result[name+"_score_prompt3"] = average_score3/len(llama2)
    average_result[name+"_score_prompt4"] = average_score4/len(llama2)
    average_result[name+"_len1_prompt1"] = average_len1/len(llama2)
    average_result[name+"_len2_prompt2"] = average_len2/len(llama2)
    average_result[name+"_len3_prompt3"] = average_len3/len(llama2)
    average_result[name+"_len4_prompt4"] = average_len4/len(llama2)
    
    #save result to jsonl
    if not os.path.exists('./diff_prompt/'+model_name.split('/')[1]):
        os.makedirs('./diff_prompt/'+model_name.split('/')[1])
        
    with open('./diff_prompt/'+model_name.split('/')[1]+'/diff_sentences_no_system_prompt_'+result_name, 'w') as f:
        json.dump(result_list, f, indent=4)
    with open('./diff_prompt/'+model_name.split('/')[1]+'/average_sentences_sentence_no_system_prompt.jsonl', 'w') as f:
        json.dump([average_result], f, indent=4)
    










