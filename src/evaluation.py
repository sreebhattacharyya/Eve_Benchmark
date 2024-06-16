import os 

os.environ["HF_HOME"] = "path/models/"

import pandas 
import torch 
from src.dataset_setup import Dataset
from src.models.eval_gpt import GPT
from src.models.eval_llava import LLaVA
from src.models.eval_blip2 import Blip2
from src.models.eval_qwenvl import QwenVL
from src.models.eval_instructblip import InstructBlip
from src.models.eval_llavanext import LlavaNext

class EvalSetup():
    def __init__(self, dataset_name, model_name, eval_protocol):
        self.dataset_name = dataset_name 
        self.model_name = model_name 
        self.eval_protocol = eval_protocol

        self.dataset = Dataset(self.dataset_name)
        self.prompt = ""
        self.model = None 

    def evaluate(self): 
        
        # unique information needed for each evaluation: 
        # label2idx, idx2label 
        # main annotaions file path -> in csv 
        # image_path relative 
        # dataframe column specifying image path 
        # dataframe column specifying label 
        # list of emotion categories in string format 

        prompt = self.get_evaluation_prompt()
        if self.model_name == "gpt": 
            self.model = GPT()
        if self.model_name == "llava": 
            self.model = LLaVA()
        if self.model_name == "blip2": 
            self.model = Blip2()
        if self.model_name == "qwen-vl": 
            self.model = QwenVL()
        if self.model_name == "instructblip":
            self.model = InstructBlip()
        if self.model_name == "llavanext":
            self.model = LlavaNext()
        
        self.model.evaluate(self.dataset, prompt)
    
    def get_evaluation_prompt(self): 

        if self.eval_protocol == "classification": 
            file_name = "path/prompts/classification_prompt.txt"
        elif self.eval_protocol == "explanation":
            file_name = "path/prompts/explanation_prompt.txt"
        elif self.eval_protocol == "context-reasoning":
            file_name = "path/prompts/context_prompt.txt"
        elif self.eval_protocol == "caption-reasoning":
            file_name = "path/prompts/caption_prompt.txt"

        with open(file_name, 'r') as f: 
            prompt = f.read()
        f.close()

        # replace the emotion categories placeholder in the prompt to include actual list of emotion categories
        prompt = prompt.replace("[emotion categories]", self.dataset.emotion_categories_string)

        return prompt
    
    def store_results(self, output_file_name):
        self.model.store_results(output_file_name)
    



        
            
            
            
