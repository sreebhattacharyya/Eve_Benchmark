from transformers import ViTFeatureExtractor, AutoImageProcessor, ViTModel, TrainingArguments, Trainer
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
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ViTModel
import torchvision
import torch.optim

from datasets import EmoSet, Emotic, FI
from model_vit import VitModel
from trainer import Trainer

PATH = "" # insert actual base directory path for use; removed here to maintain anonymity 

def main(): 

    parser = argparse.ArgumentParser(description="Choice of dataset")
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='Name of dataset: can be emotic, emoset')

    args = parser.parse_args()

    dataset_name = args.dataset_name

    image_file = None
    image_relative_path = None 
    labels = None
    dataset = None
    
    # change num_labels. Currently set to Mikels' 8.
    num_labels = 8 

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # define ViT Model to obtain image processor to be passed with dataset 
    base_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    model = VitModel(base_model, num_labels) 
    if dataset_name != "emoset":
        model.load_state_dict(torch.load(PATH + "/model_emoset.pt"))
        
    if dataset_name == "emoset":
        image_file = PATH + "/Emo-Set/data/annotations_full.csv"
        image_relative_path = "/Emo-Set/data/image/"

        dataset = EmoSet(image_file, image_relative_path, model.image_processor)
        len_dataset = dataset.__len__()
        train_len = math.ceil(0.8 * len_dataset)
        test_len = len_dataset - train_len
        print(f"Training set = {train_len}, Test set = {test_len}")
        train_ds, test_ds = random_split(dataset, [train_len, test_len])
    
    elif dataset_name == "fi":
        image_file = PATH + "/FI/annotations.csv"
        
        dataset = FI(image_file, "", model.image_processor)
        len_dataset = dataset.__len__()
        train_len = math.ceil(0.8 * len_dataset)
        test_len = len_dataset - train_len
        print(f"Training set = {train_len}, Test set = {test_len}")
        train_ds, test_ds = random_split(dataset, [train_len, test_len])
        

    # creating dataloaders for fine-tuning
    trainloader = DataLoader(train_ds, batch_size=128, num_workers=32)
    testloader = DataLoader(test_ds, batch_size=32, num_workers=32)

    # defining other utils for training 
    ce_loss = torch.nn.CrossEntropyLoss()
    weighted_loss = False # changed manually to experiment with different loss functions 

    if weighted_loss: 
        # creating weighted cross entropy loss 
        training_distribution = train_ds.__getfrequency__(return_tensors=True)
        train_dist_sum = training_distribution.sum()
        normalized_frequency = training_distribution / train_dist_sum
        normalized_frequency_reciprocal = 1 / normalized_frequency
        ce_loss = torch.nn.CrossEntropyLoss(weight=normalized_frequency_reciprocal.to(device))
    
    if dataset_name == "emoset":
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5)
    else: 
        opt = torch.optim.SGD(model.l3.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    # moving model to cuda 
    model.to(device)

    trainer = Trainer(model,
                      trainloader,
                      testloader,
                      opt,
                      scheduler,
                      ce_loss,
                      device)
    
    # define max_epochs 
    max_epochs = 30
    for current_epoch in range(max_epochs):
        print(f"Epoch {current_epoch+1}/{max_epochs}:")
        trainer.train()
    trainer.test()
    
    result_filename = "./results/" + dataset_name
    trainer.store_result(result_filename)

    model_path = "./results/model_" + dataset_name + ".pt"
    trainer.store_model(model_path)

if __name__ == '__main__': 
    main()

