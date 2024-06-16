# pip install accelerate bitsandbytes
import torch
import requests
import os 
import json 
import pandas as pd 
import numpy as np 
from PIL import Image
import PIL.Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class Blip2():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
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
                
                generated_ids = self.model.generate(
                    **inputs, 
                    max_length=512
                )
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

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
    