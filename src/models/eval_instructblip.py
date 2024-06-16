# pip install accelerate bitsandbytes
import torch
import requests
import os 
import json 
import pandas as pd 
import numpy as np 
from PIL import Image
import PIL.Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

class InstructBlip():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").to(self.device)
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
                raw_image = PIL.Image.open(image_path)
                inputs = self.processor(raw_image, prompt, return_tensors="pt").to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    min_length=1,
                    repetition_penalty=1.5
                )
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

                print(f"Generation {idx+1}: ")
                print(response)

                generation_dict = dict()
                generation_dict['prediction'] = response
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
    



