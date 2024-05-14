import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import defaultdict
import wandb

import torch
import torch.nn 
import torch.optim 
from torch.utils.data import DataLoader

import transformers
from transformers import BertTokenizer

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
EPOCHS=10

#Start a new wandb run to track this script
wandb.init(
    #Set the wandb project where this run will be logged
    project="test",

    # track hyperparameters and run metadata
    config={
    "max_len": 512,
    "train_batch_size": 4,
    "valid_batch_size": 4,
    "learning_rate": 1e-05,
    "epochs": 20,
    }
)

#Load the dataset
df = pd.read_csv("Final_Datasets/Dataset_1_test.csv")
#Adjust the labels
df.polarity=df.polarity-1

#Take a subset of data
num_polarity_1 = (df['polarity'] == 0).sum()
num_polarity_2 = (df['polarity'] == 2).sum()
sample_polarity_1 = df[df['polarity'] == 0].sample(n=10)
sample_polarity_2 = df[df['polarity'] == 1].sample(n=10)
df = pd.concat([sample_polarity_1, sample_polarity_2])
df = df.reset_index(drop=True)

#Divide test train and validation set
train_data, test_data = train_test_split(df, test_size=0.2, random_state=200)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=200)

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

#Define the optimizer for the model parameters
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

train_loss, train_accuracy=model.train_model(training_loader ,optimizer, device, EPOCHS)

model.save_model("Saved_Models/bert_sentiment_model.pth")

#emb=model.extract_pooler_output(training_loader, device)

val_loss, val_accuracy=model.evaluate_model(validation_loader, device)

print(val_loss, val_accuracy)

model.plot_loss(train_loss)
model.plot_accuracy(train_accuracy)


#Train loop
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss=0.0
#     for data in training_loader:
#         ids = data['ids'].to(device, dtype = torch.long)
#         mask = data['mask'].to(device, dtype = torch.long)
#         token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#         targets = data['targets'].to(device, dtype = torch.float)
#         targets = torch.unsqueeze(targets, dim=1)
        
#         outputs = model(ids, mask, token_type_ids)

#         #optimizer.zero_grad()
#         loss = model.loss_fn(outputs, targets)
        
#         # Accumulate the total loss
#         total_loss += loss.item()
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     #Calculate the average loss for the epoch
#     avg_loss = total_loss / len(training_loader)
#     train_loss.append(avg_loss)
    
#     #Print the average loss for the epoch
#     print(f'Epoch: {epoch}, Average Loss: {avg_loss}')
#     tr_results["epoch"]=epoch
#     tr_results["train_loss"]=avg_loss
#     # wandb.log(tr_results)
    
# def validation(epoch):
#     model.eval()
#     fin_targets=[]
#     fin_outputs=[]
#     with torch.no_grad():
#         for data in testing_loader:
#             ids = data['ids'].to(device, dtype = torch.long)
#             mask = data['mask'].to(device, dtype = torch.long)
#             token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.float)
#             targets = torch.unsqueeze(targets, dim=1)
#             outputs = model(ids, mask, token_type_ids)
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
#     return fin_outputs, fin_targets

# outputs, targets = validation(EPOCHS)
# outputs = np.array(outputs) >= 0.5
# accuracy = metrics.accuracy_score(targets, outputs)
# f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
# f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
# print(f"Accuracy Score = {accuracy}")
# print(f"F1 Score (Micro) = {f1_score_micro}")
# print(f"F1 Score (Macro) = {f1_score_macro}")

# print(targets, outputs)