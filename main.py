# incorporate everything together 
# running structure should be: python3 main.py -d dataset_name -m model_name -e evaluation_protocol 
# dataset name can be emotic, emotion6, emoset-e, emoset-m, emoset-d
# model name can be gpt4v, llava, blip2, qwen-vl
# evaluation protocol can be classification, explanation, context-reasoning, caption-reasoning

import os
import pandas 
import torch 
import argparse
from src.evaluation import EvalSetup

def main(): 
    parser = argparse.ArgumentParser(description="Choice of data, model and evaluation protocol")
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='Name of dataset: can be emotic, emotion6, emoset-e, emoset-m, emoset-d')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of model: can be gpt4v, llava, blip2, qwen-vl, llava-next')
    parser.add_argument('-e', '--eval_name', type=str, required=True, help='Name of evaluation protocol: can be classification, explanation, context-reasoning or caption-reasoning')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    eval_name = args.eval_name

    # option 1 
    # evaluator class -> dataset name, model name, evaluation protocol 
    evaluator = EvalSetup(dataset_name, model_name, eval_name)

    evaluator.evaluate()

    # for the leftovers of GPT on emotion6

    output_file_name = "/path" + dataset_name + "_" + model_name + "_" + eval_name 
    evaluator.store_results(output_file_name)
    
if __name__ == '__main__': 
    main()


