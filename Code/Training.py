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
from Evaluation_Metrics import plot_c_matrix, report_scores, plot_roc_curve

#Set hyperparmeters
MAX_LEN=512
TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=4
LEARNING_RATE=1e-05
EPOCHS=10

#Start a new wandb run to track this script
wandb.init(
    #Set the wandb project where this run will be logged
    project="new_test",

    # track hyperparameters and run metadata
    config={
    "max_len": 512,
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "learning_rate": 1e-05,
    "epochs": 10,
    }
)

#Setting up the device for GPU usage
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

#Load the dataset
df = pd.read_csv(folder_path)
#Adjust the labels
df.polarity=df.polarity-1
df.drop(columns="text")
df.dropna()

## Use this code for take a subset of data
# num_polarity_0 = (df['polarity'] == 0).sum()
# num_polarity_1 = (df['polarity'] == 1).sum()
# sample_polarity_0 = df[df['polarity'] == 0].sample(n=30)
# sample_polarity_1 = df[df['polarity'] == 1].sample(n=30)
# df = pd.concat([sample_polarity_0, sample_polarity_1])
# df = df.reset_index(drop=True)

#Divide test train and validation set
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['polarity'])
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data['polarity'])

#Reindex the dataframes
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

print("Training set size:", len(train_data))
print("Valutation set size:", len(val_data))
print("Test set size:", len(test_data))

#Initialize a BERT tokenizer from the 'bert-base-cased' pre-trained model.
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#Instantiate train and test datasets from Amazon class
training_set = AmazonTitles_Dataset(train_data, tokenizer, MAX_LEN)
validation_set=AmazonTitles_Dataset(val_data, tokenizer, MAX_LEN)
testing_set = AmazonTitles_Dataset(test_data, tokenizer, MAX_LEN)

# Configuration parameters for the DataLoader during train and test
train_params = {
    'batch_size': TRAIN_BATCH_SIZE,  
    'shuffle': True,  
    'num_workers': 0  
}
test_params = {
    'batch_size': VALID_BATCH_SIZE,  
    'shuffle': True,  
    'num_workers': 0  
}

#DataLoader for train and test
training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)
testing_loader = DataLoader(testing_set, **test_params)

#Instantiate the model from the BertClass class
model = BertClass()
#Move the model to the specified device (e.g., GPU if available)
model.to(device)

# Define the path to the saved model
save_name = "Save_Model/bert_sentiment_model_new_final.pth"
save_path = os.path.join(script_directory, os.pardir, save_name)

# Load the model's state dict (parameters) from the saved file
model.load_state_dict(torch.load(save_path)) # Use map_location=torch.device('cpu') if without GPU

#Define the optimizer for the model parameters
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Train the model
train_loss, train_accuracy, val_loss, val_accuracy=model.fit_model(training_loader, validation_loader, optimizer, device, EPOCHS, save=True)                                                                                                                                   

# Specify the name for saving the model
save_name = "Save_Model/bert_sentiment_model_final.pth"
save_path = os.path.join(script_directory, os.pardir, save_name)
# Save the model to the specified path
model.save_model(save_path)

# Plot the training and validation loss and accuracy
model.plot_loss(train_loss, val_loss)
model.plot_accuracy(train_accuracy, val_accuracy)
