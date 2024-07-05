from transformers import ViTFeatureExtractor, ViTModel, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

import os
import numpy as np 
import pandas as pd 
import torch.nn as nn
import pickle 
import argparse
import math
from matplotlib import pyplot as plt 
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from transformers import ViTModel
import torchvision
import torch.optim

from datasets import EmoSet, Emotic, FI, ArtPhoto
from model_vit import VitModel
from trainer import Trainer

PATH = "" # insert actual relative path for all data files

def main(): 

    # set seed 
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Choice of dataset, split")
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='Name of dataset: can be emotic, emoset, fi')
    parser.add_argument('-s', '--split', type=str, required=True, help='Type of split: can be train, test, all')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Enter model path to be used for evaluation. Must be .pt file.')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    model_path = args.model_path

    image_file = None
    image_relative_path = None 
    labels = None
    dataset = None
    model = None
    num_labels = 26 if dataset_name == "emotic" else 8

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # load model if path is given 
    try:
        base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        model = VitModel(base_model, num_labels)
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"Could not load model from given path: {e}")
        return

    if dataset_name == "emoset": 
        image_file = PATH + "/Emo-Set/annotations_full.csv"
        image_relative_path = "/Emo-Set/data/image/"

        dataset = EmoSet(image_file, image_relative_path, model.image_processor)
        len_dataset = dataset.__len__()

        if split == "train" or split == "test":
            train_len = math.ceil(0.8 * len_dataset)
            test_len = len_dataset - train_len 

            train_ds, test_ds = random_split(dataset, [train_len, test_len])

            dataset = train_ds if split == "train" else test_ds
    
    if dataset_name == "fi":
        image_file = PATH + "/FI/annotations.csv"
        image_relative_path = ""

        dataset = FI(image_file, image_relative_path, model.image_processor)

        if split == "train" or split == "test":
            train_len = math.ceil(0.8 * len_dataset)
            test_len = len_dataset - train_len 

            train_ds, test_ds = random_split(dataset, [train_len, test_len])

            dataset = train_ds if split == "train" else test_ds
        
    if dataset_name == "artphoto": 
        image_file = PATH + "/ArtPhoto/annotations.csv"

        dataset = ArtPhoto(image_file, "", model.image_processor)

        if split == "train" or split == "test":
            train_len = math.ceil(0.8 * len_dataset)
            test_len = len_dataset - train_len 

            train_ds, test_ds = random_split(dataset, [train_len, test_len])

            dataset = train_ds if split == "train" else test_ds
        
    # creating dataloader for evaluation 
    dataloader = DataLoader(dataset, batch_size=64, num_workers=32)

    model.to(device)

    trainer = Trainer(model, 
                      None,
                      dataloader,
                      None,
                      None,
                      None,
                      device)
    
    # starting evaluation 
    print("Beginning the Evaluation Process:")
    print("--------------------------------------------")
    print(f"Dataset Name: {dataset_name}")
    print(f"Split = {split}")
    print("--------------------------------------------")

    trainer.test()

    result_filename = result_filename = "./results/" + dataset_name + "_" + split
    trainer.store_result(result_filename)

if __name__ == '__main__':
    main()



