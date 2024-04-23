import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertModel, BertConfig


MAX_LEN=512
BATCH_SIZE=512
LEARNING_RATE=0.1 

#Set the device for PyTorch operations based on availability:
device = torch.device("cuda" if torch.cuda.is_available() 
                      else  "mps" if torch.backends.mps.is_available()
                      else "cpu"
                      )

#Initialize a BERT tokenizer from the 'bert-base-uncased' pre-trained model.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class AmazonTitles_Dataset(Dataset):
    """
    A PyTorch Dataset class for tokenizing and encoding Amazon product titles
    using a given tokenizer and preparing them for model training or inference.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing the titles and polarity labels.
        tokenizer: The tokenizer object used to tokenize and encode the titles.
        max_len (int): The maximum length of the tokenized titles after encoding.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the AmazonTitles_Dataset class.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame containing the titles and polarity labels.
            tokenizer: The tokenizer object used to tokenize and encode the titles.
            max_len (int): The maximum length of the tokenized titles after encoding.
        """
        self.tokenizer=tokenizer
        self.data=dataframe
        self.titles=dataframe.title
        self.targets=self.data.polarity
        self.max.len=max_len
    
    def __len__(self):
        """
        Returns the total number of titles in the dataset.
        """
        return len(self.titles)
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses a title and its corresponding polarity label at the given index.

        Args:
            index (int): The index of the title to retrieve.

        Returns:
            dict: A dictionary containing the tokenized and encoded title along with its attention mask,
                  token type IDs, and polarity label.

        """
        titles=str(self.titles[index])
        titles = " ".join(titles.split())
    
        inputs=self.tokenizer.encode_plus(
            titles,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids=inputs['input_ids']
        mask=inputs['attention_mask']
        token_type_ids=inputs['token_type_ids'] 

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.int)
        }
    

class BertClass(torch.nn.Module):
    """
    Neural network model based on BERT for classification tasks.
    """

    def __init__(self, dropout=0.1):
        """
        Initializes the BertClass model.

        Args:
        - dropout (float): Dropout probability (default: 0.1).
        """
        super(BertClass, self).__init__()
        # Load the pre-trained BERT model
        self.transformer = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop1 = torch.nn.Dropout(dropout)
        self.l1 = torch.nn.Linear(768, 300)
        self.act1 = torch.nn.Tanh()
        self.drop2 = torch.nn.Dropout(dropout)
        self.l2 = torch.nn.Linear(300, 100)
        self.act2 = torch.nn.Tanh()
        self.drop3 = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(100, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        """
        Defines the forward pass of the model.

        Args:
        - ids (torch.Tensor): Tensor of input token IDs.
        - mask (torch.Tensor): Tensor of attention masks.
        - token_type_ids (torch.Tensor): Tensor of token type IDs.

        Returns:
        - output (torch.Tensor): Tensor of output probabilities.
        """
        # Forward pass through the BERT model
        output_1 = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.drop1(output_1)
        output_3 = self.drop2(self.act1(self.l1(output_2)))
        output_4 = self.drop3(self.act2(self.l2(output_3)))
        output = self.out(self.l3(output_4))

        return output

def loss_fn(outputs, targets):
    """
    Computes the binary cross-entropy loss between model outputs and target labels.

    Args:
    - outputs (torch.Tensor): Tensor containing model outputs.
    - targets (torch.Tensor): Tensor containing target labels.

    Returns:
    - loss (torch.Tensor): Binary cross-entropy loss.
    """
    return torch.nn.BCELoss()(outputs, targets)

# Instantiate the BertClass model
model = BertClass()
model.to(device)

# Define the Adam optimizer with specified learning rate and model parameters
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

