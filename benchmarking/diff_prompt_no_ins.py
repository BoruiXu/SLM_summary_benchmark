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


model_name = 'Qwen/Qwen2-1.5B'

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
    result_name = llama2_files[i].split('.')[0] +'_'+ model_name.split('/')[1]+'.jsonl'

    average_score1 = 0
    average_score2 = 0
    average_len1 = 0
    average_len2 = 0
    for j in tqdm(range(len(llama2))):
        news = llama2[j]['article']
        reference = llama2[j]['qwen_reference_summary']
        
        prompt_user1 = f'News: {news}\nSummarize the news in two sentences. Summary:'
        
        messages = prompt_sys1+'\n'+prompt_user1
        
       
        model_inputs = tokenizer(messages, return_tensors="pt").to(device)
        
        stopping_criteria = stop_sequences_criteria(tokenizer, ['\n', '</s>'], model_inputs['input_ids'].shape[1], 1)

        outputs = model.generate(**model_inputs, max_new_tokens=500, stopping_criteria=stopping_criteria, do_sample=False)
        output = tokenizer.batch_decode(outputs)[0]
            
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]

        # output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response1 = clean_summary(output)
        len1 = len(response1.split(' '))
        score1 = BertScore(reference, response1)['f1'][0]*100
        print(response1)
        print("*"*30)
        prompt_user1 = f'News: {news}\nSummarize the news in two sentences. Summary:'
        messages = prompt_sys2+'\n'+prompt_user1
        
        model_inputs = tokenizer(messages, return_tensors="pt").to(device)
        
        stopping_criteria = stop_sequences_criteria(tokenizer, ['\n', '</s>'], model_inputs['input_ids'].shape[1], 1)

        outputs = model.generate(**model_inputs, max_new_tokens=500, stopping_criteria=stopping_criteria, do_sample=False)
        output = tokenizer.batch_decode(outputs)[0]
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]

        # output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
    with open('./diff_prompt/phi2/'+result_name, 'w') as f:
        json.dump(result_list, f, indent=4)
    with open('./diff_prompt/phi2/average.jsonl', 'w') as f:
        json.dump([average_result], f, indent=4)
    










