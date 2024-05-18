import numpy as np
import pandas as pd
import wandb
import os
from matplotlib import pyplot as plt

import torch
import torch.nn 
import torch.optim 
from torch.utils.data import DataLoader

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import transformers
from transformers import BertTokenizer

from Model import BertClass, AmazonTitles_Dataset
from Evaluation_Metrics import plot_c_matrix, report_scores

# Set hyperparameters
MAX_LEN = 512
VALID_BATCH_SIZE = 4

# Setting up the device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() 
                      else  "mps" if torch.backends.mps.is_available()
                      else "cpu"
                      )

# Get the absolute path of the current script file
script_path = os.path.abspath(__file__)
# Get the directory path containing the script file
script_directory = os.path.dirname(script_path)
folder_name = "Final_Datasets/Dataset_1_train.csv"
# Create the complete path using os.path.join() and os.pardir to "go back" one folder
folder_path = os.path.join(script_directory, os.pardir, folder_name)

# Load the dataset
df = pd.read_csv(folder_path)
# Adjust the labels
df.polarity = df.polarity - 1
df.drop(columns="text")
df.dropna()

# Divide the dataset into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['polarity'])
# Reset the index of the test dataset
test_data = test_data.reset_index(drop=True)

# Initialize a BERT tokenizer from the 'bert-base-cased' pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Instantiate the test dataset using the AmazonTitles_Dataset class
testing_set = AmazonTitles_Dataset(test_data, tokenizer, MAX_LEN)

# Set the parameters for the DataLoader
test_params = {
    'batch_size': VALID_BATCH_SIZE,  
    'shuffle': True,  
    'num_workers': 0  
}

# Create the DataLoader for testing
testing_loader = DataLoader(testing_set, **test_params)

# Instantiate the BERT model
model = BertClass()

# Define the path to the saved model
save_name = "Save_Model/bert_sentiment_model_new_final.pth"
save_path = os.path.join(script_directory, os.pardir, save_name)

# Load the model's state dict (parameters) from the saved file
model.load_state_dict(torch.load(save_path)) # Use map_location=torch.device('cpu') if without GPU

# Test the model on the test dataset
test_loss, test_accuracy, all_predictions, all_targets = model.test_model(testing_loader, device)

report_scores(all_targets, all_predictions)
plot_c_matrix(all_targets, all_predictions, "our LLM classifier")
