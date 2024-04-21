import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertModel, BertConfig

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