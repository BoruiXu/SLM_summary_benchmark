#this noetbook focus on the evaluation of summary quality using rouge and bert score
from datasets import load_dataset
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import json
# Set OpenAI's API key and API base to use vLLM's API server.
from scipy.stats import kendalltau, spearmanr
from openai import OpenAI
import sacrebleu
#load bert score model
from rouge_score import rouge_scorer, scoring
from evaluate import load
bert_score = load("bertscore")



def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68
    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge4", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    
    return {type: result[type].mid.fmeasure  for type in rouge_types}


def BertScore(refs, preds):
    bert_score_res = bert_score.compute(predictions=[refs], references=[preds], model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    return bert_score_res

def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(preds, refs, smooth_method="exp", smooth_value=0.0, force=False,
                                  lowercase=False, tokenize="intl", use_effective_order=False).score
    return score

def get_score(refs, preds,metric):
    
    result = 0
    if(metric[:5]=="rouge"):
        rouge_res = rouge([refs], [preds])
        result = rouge_res[metric]
    elif(metric=="bertscore"):
        result = BertScore(refs, preds)["f1"][0]
    elif(metric=="bleu"):
        result = bleu([refs], [preds])
    elif(metric=="chrf"):
        if(preds==""):
            preds = " "
        result = sacrebleu.corpus_chrf(preds, [refs]).score
    
    
    return result


def loop_score_api_chat(summary_list, reference_list, metric):
    
    score_list = []
    
    for i in range(len(summary_list)):
            reference = reference_list[i]
            summary = summary_list[i]
            score_list.append(get_score(reference, summary, metric))
    
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
    


def evaluate(path, aspect, metric = "rougeLsum", reference_model = 'reference', few_shot = 0):
    
    dataset_name = path.split("/")[-1].split(".")[0]
    print("evaluating dataset: ", dataset_name)
    model_name = metric
    
    target_dataset = pd.read_json(path)
    model_list = list(set(target_dataset['model'].tolist()))
    model_list.remove(reference_model)
    model_list = sorted(model_list)
    
    reference = target_dataset[target_dataset["model"]==reference_model]
    
    
    #save result    
    model_eva_dict= {}
    human_eva_dict = {}

    for m in model_list:
        print("evaluating model: ", m)
        tmp_dataset = target_dataset[(target_dataset['model']==m )]
        
        tmp_news_list = tmp_dataset['article'].tolist()
        tmp_summary_list = tmp_dataset['summary'].tolist()
        tmp_score_list = tmp_dataset[aspect].tolist()
        
        tmp_reference_list = []
        
        
        for i in range(len(tmp_news_list)):
            tmp_reference_list.append(reference[reference['article']==tmp_news_list[i]]['summary'].values[0])
        
        
        score_list = loop_score_api_chat(tmp_summary_list, tmp_reference_list, metric)
        
        model_eva_dict[m] = score_list
        human_eva_dict[m] = tmp_score_list  
    
    #save the result
    
    save_name = str(model_name)+'_'+str(dataset_name)+'_'+str(aspect)+'_eva.json'
    human_save = 'human_score_'+str(dataset_name)+'_'+str(aspect)+'_eva.json'
   
    
    # with open('./LLM_evaluation_correlation_with_human/'+save_name, 'w') as fp:
    #     json.dump(model_eva_dict, fp)
    # with open('./LLM_evaluation_correlation_with_human/'+human_save, 'w') as fp:
    #     json.dump(human_eva_dict, fp)
    

    correlation_score(model_eva_dict, human_eva_dict)
    
if __name__ == "__main__":
    
    p = '/home/xbr/LLM/benchmark_llm_summarization/likert_evaluation_results_cnndm_average.json'#'./filter_annotations_summeval.jsonl'#'./filter_annotations_summeval.jsonl'# #
    aspect = "relevance"
    evaluate(p, aspect)
