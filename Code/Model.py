import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertModel, BertConfig


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
        Initializes the AmazonTitles_Dataset.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame containing the titles and polarity labels.
            tokenizer: The tokenizer object used to tokenize and encode the titles.
            max_len (int): The maximum length of the tokenized titles after encoding.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.titles = dataframe.title
        self.targets = self.data.polarity
        self.max_len = max_len
    
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.titles)
    
    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing input ids, attention mask, token type ids, and target labels.
        """
        titles = str(self.titles[index])
        titles = " ".join(titles.split())
    
        inputs = self.tokenizer.encode_plus(
            titles,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids'] 

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }   
        

class BertClass(torch.nn.Module):
    """
    A PyTorch module defining a neural network architecture for sentiment analysis using BERT.
    """

    def __init__(self, dropout=0.1):
        """
        Initializes the BertClass.

        Args:
            dropout (float): Dropout probability.
        """
        super(BertClass, self).__init__()
        self.transformer = transformers.BertModel.from_pretrained('bert-base-cased')
        self.batch1 = torch.nn.BatchNorm1d(768)  # Batch normalization layer
        #self.drop1 = torch.nn.Dropout(dropout)  # Dropout layer
        self.l1 = torch.nn.Linear(768, 300)  # First linear layer
        self.act1 = torch.nn.ReLU()  # ReLU activation function
        #self.batch2 = torch.nn.BatchNorm1d(300)  # Batch normalization layer
        self.drop2 = torch.nn.Dropout(dropout)  # Dropout layer
        self.l2 = torch.nn.Linear(300, 100)  # Second linear layer
        self.act2 = torch.nn.ReLU()  # ReLU activation function
        #self.batch3 = torch.nn.BatchNorm1d(100)  # Batch normalization layer
        self.drop3 = torch.nn.Dropout(dropout)  # Dropout layer
        self.l3 = torch.nn.Linear(100, 1)  # Output linear layer
        self.output=torch.nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        """
        Defines the forward pass of the neural network.

        Args:
            ids (torch.Tensor): Input token ids.
            mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type ids.

        Returns:
            torch.Tensor: Model output.
        """
        _, output_1 = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.batch1(output_1)
        output_3 = self.drop2(self.act1(self.l1(output_2)))
        output_4 = self.drop3(self.act2(self.l2(output_3)))
        #output = self.l3(output_4)
        output=self.output(self.l3(output_4))

        return output

    def loss_fn(self, outputs, targets):
        """
        Computes the loss between model predictions and target labels.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Loss value.
        """
        return torch.nn.BCELoss()(outputs, targets)



