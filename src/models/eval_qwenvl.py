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
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import requests
import PIL.Image

# import autogen
# from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
# from autogen.agentchat.contrib.llava_agent import LLaVAAgent, llava_call
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# from transformers import BitsAndBytesConfig
from transformers import pipeline
import torch

class QwenVL():
    def __init__(self):
        self.model_id = "Qwen/Qwen-VL-Chat"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self.result = []

    def evaluate(self, dataset, prompt):

        # read dataset dataframe 
        df = pd.read_csv(dataset.annotations_path)

        print("Beginning the evaluation processs:")

        for idx, row in df.iterrows():
            
            # get image path 
            image_path = dataset.relative_image_path + row[dataset.image_path_col]
            if not os.path.exists(image_path):
                continue

            label = row[dataset.label_col] if dataset.label_string else dataset.idx2label[row[dataset.label_col]]
            
            try:
                query = self.tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': prompt}
                ])
                
                response, _ = self.model.chat(self.tokenizer, query=query, history=None)
                
                # get JSON output from the response
                if '{' not in response: 
                    print(f"Generation {idx+1}: {response}")
                    continue

                generation_json = response.split("{")[1]
                generation_json = '{' + generation_json

                print(f"Generation {idx+1}: ")
                print(generation_json)

                generation_dict = json.loads(generation_json)
                generation_dict['ground_truth'] = label
                generation_dict['image'] = image_path
                self.result.append(generation_dict)
            
            except Exception as e:
                print(f"Generation {idx+1} skipped because of {e}")
                print()

    def store_results(self, output_file_name):

        # storing results from list of dict into given file name 
        with open(output_file_name, 'w') as outfile: 
            json.dump(self.result, outfile, indent=4)

