from transformers import ViTFeatureExtractor, ViTModel, TrainingArguments, Trainer
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import os
import numpy as np 
import pandas as pd 
import torch.nn as nn
import pickle 
import argparse
from matplotlib import pyplot as plt 
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data
import torchvision
from datasets import EmoSet

class VitModel(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super(VitModel, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", return_tensors='pt')
        self.vit = base_model
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.num_labels = num_labels
        self.dropout1 = torch.nn.Dropout(0.2)
        self.l1 = torch.nn.Linear(self.vit.config.hidden_size, self.vit.config.hidden_size//2)
        self.relu1 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(self.vit.config.hidden_size//2, self.num_labels*2)
        self.relu2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(self.num_labels*2, self.num_labels)

        
    def forward(self, pixel_values): 
        
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout1(outputs.last_hidden_state[:,0])
        output = self.l1(output)
        output = self.relu1(output)
        output = self.dropout2(output)
        output = self.l2(output)
        output = self.relu2(output)
        logits = self.l3(output)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return probs
