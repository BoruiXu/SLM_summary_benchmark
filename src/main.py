from util import setup_parser, eval_logger, simple_parse_args_string
import logging
from load_data import load_data
import generate
import json

def main():
    args = setup_parser().parse_args()
    
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    
    model = args.model
    task = args.tasks
    prompt = args.prompt
    model_args = args.model_args
    num_fewshot = args.num_fewshot
    batch_size = args.batch_size
    dataset_path = args.dataset
    device = args.device
    gen_args_list = simple_parse_args_string(args.gen_kwargs)
    output_path = args.output_path
    prompt = args.prompt
    
    #load data to process
    # data = load_data('/home/xbr/LLM/evaluation_summary_model/sample_dataset_Qwen_summary/newsroom_sample_500_0k5_1k5_training_set.json')
    dataset = load_data(dataset_path)
    # print(data.shape)
    
    
    if(task == "generate" ):
        eval_logger.info("generate summary using model "+model_args)
        
        if(prompt is None):
            prompt = "You are a helpful summary assistant. You can help users summarize articles in two sentences."
        
        result = generate.generate_summary(
            model,
            model_args,
            dataset,
            num_fewshot,
            batch_size,
            prompt,
            device,
            gen_args_list)

        #save result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        print("generation done")
        
    elif(task == "evaluate"):
        eval_logger.info("evaluate generated summary using model "+model_args)
        
        if(prompt is None):
            prompt = "You are a helpful summary assistant. You can help users evaluate the relevance of generated summarize.\
                The score of relevance is form 1 to 5, 5 is the best, 1 is the worst.\n\
                Only give me the score."
        
                
                
        
    else:
        eval_logger.error("Invalid task type")
        
    
    
    

if __name__ == "__main__":
   
    main()

