import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertModel, BertConfig

from matplotlib import pyplot as plt
import wandb



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
        self.classifier=torch.nn.Sequential(
            torch.nn.BatchNorm1d(768),
            torch.nn.Linear(768, 300),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )
        
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
        output_1 = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
        output=self.classifier(output_1["pooler_output"])
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
    
    def save_model(self, path):
        """
        Saves the model state to a file.

        Args:
            path (str): Path to save the model state.
        """
        torch.save(self.state_dict(), path)
        
    def train_model(self, train_loader, optimizer, device, num_epochs):
        """
        Trains the model using the specified data loader, optimizer, and device.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
            num_epochs (int): Number of epochs for training.

        Returns:
            list, list: Training losses, training accuracies.
        """
        # Vectors for loss and accuracy
        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs):
            
            self.train()
            
            # Freeze transformer parameters for the first 7 epochs
            if epoch < (num_epochs-3):
                for param in self.transformer.parameters():
                    param.requires_grad = False
            # Unfreeze transformer parameters for the last 3 epochs
            else:
                for param in self.transformer.parameters():
                    param.requires_grad = True
            
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for data in train_loader:
                # Take inputs and targets
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                targets = torch.unsqueeze(targets, dim=1)

                # Zero gradients for every batch!
                optimizer.zero_grad()
                
                # Make predictions for this batch
                outputs = self(ids, mask, token_type_ids)
                
                # Compute loss
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                
                # Adjust learning weights
                optimizer.step()

                total_loss += loss.item()

                # Compute accuracies
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)

            accuracy = correct_predictions / total_predictions
            avg_loss=total_loss / len(train_loader.dataset)       
            print(f'Epoch: {epoch}, Average Loss: {avg_loss}')
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

        return train_losses, train_accuracies
    
    def evaluate_model(self, val_loader, device):
        """
        Evaluates the model on the validation data.

        Args:
            val_loader (DataLoader): DataLoader for validation or test data.
            device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').

        Returns:
            float, float: Validation loss, validation accuracy.
        """
        
        # Set the model to evaluation mode
        self.eval()

        # Initialize variables for loss and accuracy
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Turn off gradient computation
        with torch.no_grad():
            for data in val_loader:
                # Take inputs and targets
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                targets = torch.unsqueeze(targets, dim=1)

                # Make predictions for this batch
                outputs = self(ids, mask, token_type_ids)

                # Compute loss
                loss = self.loss_fn(outputs, targets)

                # Accumulate validation loss
                val_loss += loss.item()

                # Compute accuracies
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)

        # Calculate validation accuracy and loss
        val_accuracy = correct_predictions / total_predictions
        avg_val_loss = val_loss / len(val_loader.dataset)

        return avg_val_loss, val_accuracy

    
    def plot_loss(self, train_losses, val_losses=None, figsize=(8,6), print_every=1):
        """
        Plot training and validation losses.

        Args:
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses (optional).
            figsize (tuple): Size of the figure.
            print_every (int): Interval for printing training progress.
        """
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_losses, label='Train Loss')
        if val_losses:
            plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.xticks(epochs[::print_every])
        plt.show()
        
    def plot_accuracy(self, train_accuracies, val_accuracies=None, figsize=(8,6), print_every=1):
        """
        Plot training and validation accuracies.

        Args:
            train_accuracies (list): List of training accuracies.
            figsize (tuple): Size of the figure.
            val_accuracies (list): List of validation accuracies (optional).
            print_every (int): Interval for printing training progress.
        """
        epochs = range(1, len(train_accuracies) + 1)
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        if val_accuracies:
            plt.plot(epochs, val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies')
        plt.legend()
        plt.xticks(epochs[::print_every])
        plt.show()



