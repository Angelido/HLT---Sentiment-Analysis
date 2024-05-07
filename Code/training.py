import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertModel, BertConfig

from Model import BertClass, AmazonTitles_Dataset

#Setting up the device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() 
                      else  "mps" if torch.backends.mps.is_available()
                      else "cpu"
                      )

#Set hyperparmeters
MAX_LEN=512
TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=4
LEARNING_RATE=1e-05
EPOCHS=20

#Load the dataset
df = pd.read_csv("../Final_Datasets/Dataset_1_test.csv")
#Adjust the labels
df.polarity=df.polarity-1

#Take a subset of data
num_polarity_1 = (df['polarity'] == 0).sum()
num_polarity_2 = (df['polarity'] == 2).sum()
sample_polarity_1 = df[df['polarity'] == 0].sample(n=10)
sample_polarity_2 = df[df['polarity'] == 1].sample(n=10)
df = pd.concat([sample_polarity_1, sample_polarity_2])

#Divide test and train set
train_size=0.8
train_dataset=df.sample(frac=train_size, random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

#Initialize a BERT tokenizer from the 'bert-base-cased' pre-trained model.
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#Instantiate train and test datasets from Amazon class
training_set = AmazonTitles_Dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = AmazonTitles_Dataset(test_dataset, tokenizer, MAX_LEN)

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
testing_loader = DataLoader(testing_set, **test_params)

#Instantiate the model from the BertClass class
model = BertClass()
#Move the model to the specified device (e.g., GPU if available)
model.to(device)

#Define the optimizer for the model parameters
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    total_loss=0.0
    for data in training_loader:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        targets = torch.unsqueeze(targets, dim=1)
        
        outputs = model(ids, mask, token_type_ids)

        #optimizer.zero_grad()
        loss = model.loss_fn(outputs, targets)
        
        # Accumulate the total loss
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #Calculate the average loss for the epoch
    avg_loss = total_loss / len(training_loader)

    #Print the average loss for the epoch
    print(f'Epoch: {epoch}, Average Loss: {avg_loss}')
    
for epoch in range(EPOCHS):
    train(epoch)