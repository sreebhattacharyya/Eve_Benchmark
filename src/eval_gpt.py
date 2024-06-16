# pip install accelerate bitsandbytes
import torch
import requests
import os 
import json 
import pandas as pd 
import numpy as np 
from PIL import Image
import PIL.Image
import base64

from openai import OpenAI

class GPT():
    def __init__(self):

        # create client to be used for completions 
        self.client = OpenAI()

        # set the model name 
        self.model_id = "gpt-4o"
    
        self.result = []

    def encode_image(self, image_path):

        # Function to encode the image
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def evaluate(self, dataset, prompt): 
        
        # read dataset dataframe 
        df = pd.read_csv(dataset.annotations_path)

        print("Beginning the evaluation process:")
        print()

        for idx, row in df.iterrows():
            
            # get image path 
            image_path = dataset.relative_image_path + row[dataset.image_path_col]
            if not os.path.exists(image_path):
                continue 

            label = row[dataset.label_col] if dataset.label_string else dataset.idx2label[row[dataset.label_col]]

            # convert image to base64 encoded format
            image_base64 = self.encode_image(image_path)

            try: 
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    response_format={ "type": "json_object" },
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "low"
                                }
                            },
                        ],
                        }
                    ],
                    max_tokens=160,
                    )
                
                generation_json = response.choices[0].message.content

                print(f"Generation {idx+1}:")
                print(generation_json)

                generation_dict = json.loads(generation_json)
                generation_dict["ground_truth"] = label
                generation_dict["image"] = image_path

                self.result.append(generation_dict)

            except Exception as e: 
                print(f"Generation {idx+1} failed due to following exception: {e}")

    def store_results(self, output_file_name):
        
        with open(output_file_name, 'w') as outfile:
            json.dump(self.result, outfile, indent=4)