import torch 
import pandas as pd 
import numpy as np 
from time import sleep
import string
import requests
import subprocess

# import things for running autogen library 
import json
import os
os.environ["HF_HOME"] = "/scratch/bbmr/sbhattacharyya1/models/"
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import requests
import PIL.Image

# import autogen
# from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
# from autogen.agentchat.contrib.llava_agent import LLaVAAgent, llava_call
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.generation import GenerationConfig
# from transformers import BitsAndBytesConfig
from transformers import pipeline
import torch

class LlavaNext():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "llava-hf/llava-v1.6-34b-hf"
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id, cache_dir="/scratch/bbmr/sbhattacharyya1/models/")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=True)
        self.result = []

    def evaluate(self, dataset, prompt):

        # read dataset dataframe 
        df = pd.read_csv(dataset.annotations_path)

        # editing the prompt
        if "vicuna" in self.model_id:
            prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed answers to the human's questions. USER: <image>\n" + prompt + "\nASSISTANT:"
        elif "mistral" in self.model_id: 
            prompt = "[INST] <image>\n" + prompt + " [/INST]"
        elif "34b" in self.model_id: 
            prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n"+ prompt + "<|im_end|><|im_start|>assistant\n"

        print("Beginning the evaluation processs:")

        for idx, row in df.iterrows():
            
            # get image path 
            image_path = dataset.relative_image_path + row[dataset.image_path_col]
            if not os.path.exists(image_path):
                continue

            label = row[dataset.label_col] if dataset.label_string else dataset.idx2label[row[dataset.label_col]]
            
            try:
                raw_image = PIL.Image.open(image_path)
                inputs = self.processor(prompt, raw_image, return_tensors="pt").to(self.device)
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=512
                )
                response = self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()
                
                # get JSON output from the response
                if '{' not in response: 
                    print(f"Generation {idx+1} not in JSON format: {response}")
                    continue

                generation_json_list = response.split("{")
                generation_json = generation_json_list[1]
                if len(generation_json) > 2: 
                    # multiple '{' was found in the response. Choose the second set.
                    generation_json = generation_json_list[2]
                
                generation_json = '{' + generation_json

                checked_json = ""
                for s in generation_json: 
                    checked_json = checked_json + s 
                    if s == '}':
                        break

                print(f"Generation {idx+1}: ")
                print(checked_json)

                generation_dict = json.loads(checked_json)
                generation_dict['ground_truth'] = label
                generation_dict['image'] = image_path
                self.result.append(generation_dict)
            
            except Exception as e:
                print(f"Generation {idx+1} skipped because of {e}")
                print()

    def store_results(self, output_file_name):
        # appending the filename with model version 
        model_version = None
        if "vicuna" in self.model_id:
            if "7b" in self.model_id:
                model_version = "_vicuna7b"
            elif "13b" in self.model_id:
                model_version = "_vicuna13b"
        elif "mistral" in self.model_id:
            model_version = "_mistral7b"
        elif "34b" in self.model_id: 
            model_version = "_34b"

        output_file_name += model_version

        # storing results from list of dict into given file name 
        with open(output_file_name, 'w') as outfile: 
            json.dump(self.result, outfile, indent=4)