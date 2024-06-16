import os 

os.environ["HF_HOME"] = "path/models"

import pandas 
import torch

class Dataset(): 
    def __init__(self, dataset_name): 
        self.dataset_name = dataset_name

        # unique information needed for each evaluation: 
        # label2idx, idx2label 
        # main annotaions file path -> in csv 
        # image_path relative 
        # dataframe column specifying image path 
        # dataframe column specifying label 
        # whether label in dataframe is index or string 
        # list of emotion categories in string format 

        self.label2idx = []
        self.idx2label = []
        self.annotations_path = None 
        self.relative_image_path = None
        self.image_path_col = None
        self.label_col = None
        self.label_string = False # is the data label a string
        self.emotion_categories_string = None 

        if dataset_name == "emotion6":
            self.populate_emotion6()
        elif dataset_name == "emoset":
            self.populate_emoset()
        elif dataset_name == "artphoto":
            self.populate_artphoto()
        elif dataset_name == "abstract":
            self.populate_abstract()
        elif dataset_name == "fi":
            self.populate_fi()
            

    def populate_emotion6(self): 

        self.label2idx = {
            'anger': 0,
            'disgust': 1,
            'fear': 2, 
            'joy': 3,
            'sadness': 4, 
            'surprise': 5, 
            'neutral': 6
        }

        self.idx2label = {
            0: 'anger',
            1: 'disgust', 
            2: 'fear', 
            3: 'joy', 
            4: 'sadness', 
            5: 'surprise',
            6: 'neutral'
        }

        self.annotations_path = "path/Emotion6/annotations.csv"

        self.relative_image_path = "path/Emotion6/images/"

        self.image_path_col = 'file_path'

        self.label_col = 'label'

        self.label_string = False 

        self.emotion_categories_string = ', '.join(list(self.label2idx.keys()))
    

    def populate_emoset(self):

        self.label2idx = {"amusement": 0,
            "anger": 1, 
            "awe": 2, 
            "contentment": 3, 
            "disgust": 4,
            "excitement": 5, 
            "fear": 6,
            "sadness": 7
        }
        
        self.idx2label = {0: "amusement",
                    1: "anger", 
                    2: "awe", 
                    3: "contentment", 
                    4: "disgust", 
                    5: "excitement", 
                    6: "fear", 
                    7: "sadness"
                }
        
        # the default benchmark contains only the moderate and difficult samples
        # to additionally evaluate on the easy sample, use 'annotations_easy.csv'
        self.annotations_path = "path/Emo-Set/data/annotations_moderate.csv"

        self.relative_image_path = "path/Emo-Set/data/image/"
        
        self.image_path_col = 'img_id'

        self.label_col = 'ground_truth'

        self.label_string = False 

        self.emotion_categories_string = ', '.join(list(self.label2idx.keys()))


    def populate_artphoto(self):
        self.label2idx = {"amusement": 0,
            "anger": 1, 
            "awe": 2, 
            "contentment": 3, 
            "disgust": 4,
            "excitement": 5, 
            "fear": 6,
            "sadness": 7
        }

        self.idx2label = {0: "amusement",
                    1: "anger", 
                    2: "awe", 
                    3: "contentment", 
                    4: "disgust", 
                    5: "excitement", 
                    6: "fear", 
                    7: "sadness"
                }
        
        self.annotations_path = "path/ArtPhoto/annotations.csv"

        self.relative_image_path = ""
        
        self.image_path_col = 'image_path'

        self.label_col = 'label'

        self.label_string = True 

        self.emotion_categories_string = ', '.join(list(self.label2idx.keys()))


    def populate_abstract(self):
        self.label2idx = {"amusement": 0,
            "anger": 1, 
            "awe": 2, 
            "content": 3, 
            "disgust": 4,
            "excitement": 5, 
            "fear": 6,
            "sad": 7
        }

        self.idx2label = {0: "amusement",
                    1: "anger", 
                    2: "awe", 
                    3: "content", 
                    4: "disgust", 
                    5: "excitement", 
                    6: "fear", 
                    7: "sad"
                }
        
        self.annotations_path = "path/Abstract/annotations.csv"

        self.relative_image_path = ""
        
        self.image_path_col = 'image_path'

        self.label_col = 'label'

        self.label_string = True 

        self.emotion_categories_string = ', '.join(list(self.label2idx.keys()))
        

    def populate_fi(self):

        self.label2idx = {"amusement": 0,
            "anger": 1, 
            "awe": 2, 
            "contentment": 3, 
            "disgust": 4,
            "excitement": 5, 
            "fear": 6,
            "sadness": 7
        }
        
        self.idx2label = {0: "amusement",
                    1: "anger", 
                    2: "awe", 
                    3: "contentment", 
                    4: "disgust", 
                    5: "excitement", 
                    6: "fear", 
                    7: "sadness"
                }
        
        self.annotations_path = "path/FI/annotations_hard.csv"

        self.relative_image_path = ""
        
        self.image_path_col = 'img_id'

        self.label_col = 'ground_truth'

        self.label_string = False 

        self.emotion_categories_string = ', '.join(list(self.label2idx.keys()))
