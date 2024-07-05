import os
import numpy as np 
import pandas as pd 
import torch.nn as nn
import pickle 
from matplotlib import pyplot as plt 
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from PIL import Image
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import torch.utils.data
import torchvision

class EmoSet(Dataset):
  def __init__(self, img_file, image_relative_path, pixel_transform):
    self.img_file = img_file
    self.img_df = pd.read_csv(img_file, index_col=False)
    self.img_relative_path = image_relative_path
    
    print(f"Information in the image dataset = {self.img_df.columns.values}")
    
    self.img_labels = self.img_df.loc[:, "emotion"]

    self.label2idx = self.get_label2idx()
    
    print("Label to Index Dictionary for Dataset: ")
    print(self.label2idx)
  
    self.transform = pixel_transform

  def get_label2idx(self):
    return {"amusement": 0,
            "anger": 1, 
            "awe": 2, 
            "contentment": 3, 
            "disgust": 4,
            "excitement": 5, 
            "fear": 6,
            "sadness": 7}

  def __len__(self):
    return len(self.img_labels.values)

  def __getitem__(self, idx):
    img_row = self.img_df.iloc[idx]
    
    img_id = img_row['image_id']
    label = img_row['emotion']
    # creating complete image path 
    img_path = self.img_relative_path + label + "/" + img_id + ".jpg"
    
    image = Image.open(img_path).convert('RGB')

    pixel_values = self.transform(image)['pixel_values']
    pixel_values = torch.tensor(np.array(pixel_values)).squeeze(0)

    label = img_row['emotion']
    if not self.label2idx is None:
      label = self.label2idx[label] 
    
    return pixel_values, label, img_id
  
  def __getlabels__(self):
    return self.img_labels.values


  def print_info(self):
    print("Labels:")
    print()
    print(self.img_labels.value_counts())
    print("Dataframe: ")
    print()
    print(self.img_df.head())


class FI(Dataset): 
  def __init__(self, img_file, image_relative_path, pixel_transform):
    self.img_file = img_file
    self.img_df = pd.read_csv(img_file, index_col=False)
    self.img_relative_path = image_relative_path
    
    print(f"Information in the image dataset = {self.img_df.columns.values}")
    
    self.img_labels = self.img_df.loc[:, "label"]

    self.label2idx = self.get_label2idx()
    
    print("Label to Index Dictionary for Dataset: ")
    print(self.label2idx)
  
    self.transform = pixel_transform

  def get_label2idx(self):
    return {"amusement": 0,
            "anger": 1, 
            "awe": 2, 
            "contentment": 3, 
            "disgust": 4,
            "excitement": 5, 
            "fear": 6,
            "sadness": 7}

  def __len__(self):
    return len(self.img_labels.values)

  def __getitem__(self, idx):
    img_row = self.img_df.iloc[idx]
    
    img_id = img_row['image_path']
    label = img_row['label']
    # creating complete image path 
    img_path = self.img_relative_path + img_id
    
    image = Image.open(img_path).convert('RGB')

    pixel_values = self.transform(image)['pixel_values']
    pixel_values = torch.tensor(np.array(pixel_values)).squeeze(0)

    label = img_row['label']
    if not self.label2idx is None:
      label = self.label2idx[label] 
    
    return pixel_values, label, img_id
  
  def __getlabels__(self):
    return self.img_labels.values


  def print_info(self):
    print("Labels:")
    print()
    print(self.img_labels.value_counts())
    print("Dataframe: ")
    print()
    print(self.img_df.head())
