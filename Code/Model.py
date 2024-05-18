import torch
import torch.nn 
from torch.utils.data import Dataset

import transformers

from matplotlib import pyplot as plt
import wandb
import pandas as pd
import numpy as np
from collections import defaultdict



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
            torch.nn.BatchNorm1d(300),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(300, 100),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(100),
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
        
        
    def fit_model(self, train_loader, val_loader, optimizer, device, num_epochs, save=False):
        """
        Train and validate the model using the specified data loaders, optimizer, and device.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
            num_epochs (int): Number of epochs for training.
            save (bool, optional): If True, save the model during training. Default is False.

        Returns:
            list, list, list, list: Training losses, training accuracies, validation losses, validation accuracies.
        """
        # Vectors for loss and accuracy
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Dictionaries for storing results for WandB logging
        tr_results = defaultdict(list)
        val_results = defaultdict(list)

        for epoch in range(num_epochs):
            
            # Set the model to training mode
            self.train()
            
            # Freeze transformer parameters for the first 7 epochs
            if epoch < (num_epochs-3):
                for param in self.transformer.parameters():
                    param.requires_grad = False
            # Unfreeze transformer parameters for the last 3 epochs
            else:
                for param in self.transformer.parameters():
                    param.requires_grad = True
            
            # Training phase
            total_train_loss = 0.0
            correct_train_predictions = 0
            total_train_predictions = 0
            
            for train_data in train_loader:
                # Take inputs and targets
                ids = train_data['ids'].to(device, dtype=torch.long)
                mask = train_data['mask'].to(device, dtype=torch.long)
                token_type_ids = train_data['token_type_ids'].to(device, dtype=torch.long)
                targets = train_data['targets'].to(device, dtype=torch.float)
                targets = torch.unsqueeze(targets, dim=1)

                # Zero gradients for every batch
                optimizer.zero_grad()
                
                # Make predictions for this batch
                outputs = self(ids, mask, token_type_ids)
                
                # Compute loss
                train_loss = self.loss_fn(outputs, targets)
                train_loss.backward()
                
                # Adjust learning weights
                optimizer.step()

                total_train_loss += train_loss.item()

                # Compute accuracies
                train_predictions = (outputs > 0.5).float()
                correct_train_predictions += (train_predictions == targets).sum().item()
                total_train_predictions += targets.size(0)
                
                # Save the model if 'save' is True (for long training)
                if save:
                    save_path=("../Save_Model/bert_sentiment_model_new_final.pth")
                    self.save_model(save_path)
            
            # Calculate average training loss and accuracy
            train_accuracy = correct_train_predictions / total_train_predictions
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            self.eval()
            
            total_val_loss = 0.0
            correct_val_predictions = 0
            total_val_predictions = 0
            
            with torch.no_grad():
                for val_data in val_loader:
                    # Take inputs and targets
                    ids = val_data['ids'].to(device, dtype=torch.long)
                    mask = val_data['mask'].to(device, dtype=torch.long)
                    token_type_ids = val_data['token_type_ids'].to(device, dtype=torch.long)
                    targets = val_data['targets'].to(device, dtype=torch.float)
                    targets = torch.unsqueeze(targets, dim=1)

                    # Make predictions for this batch
                    val_outputs = self(ids, mask, token_type_ids)
                    
                    # Compute loss
                    val_loss = self.loss_fn(val_outputs, targets)
                    total_val_loss += val_loss.item()

                    # Compute accuracies
                    val_predictions = (val_outputs > 0.5).float()
                    correct_val_predictions += (val_predictions == targets).sum().item()
                    total_val_predictions += targets.size(0)

            # Calculate average validation loss and accuracy
            val_accuracy = correct_val_predictions / total_val_predictions
            avg_val_loss = total_val_loss / len(val_loader)

            # Print training and validation metrics
            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss}, Train Accuracy: {train_accuracy}')
            print(f'Epoch: {epoch}, Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy}')

            # Append metrics to lists
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Log metrics to WandB
            tr_results["epoch"] = epoch
            tr_results["train_loss"] = avg_train_loss
            tr_results["train_accuracy"] = train_accuracy
            wandb.log(tr_results)
            
            val_results["epoch"] = epoch
            val_results["val_loss"] = avg_val_loss
            val_results["val_accuracy"] = val_accuracy
            wandb.log(val_results)

        return train_losses, train_accuracies, val_losses, val_accuracies

    
    def test_model(self, test_loader, device):
        """
        Evaluates the model on the test data.

        Args:
            test_loader (DataLoader): DataLoader for test data.
            device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').

        Returns:
            float, float, list, list: Test loss, test accuracy, all predictions, all targets., float: Validation loss, validation accuracy.
        """
        
        # Set the model to evaluation mode
        self.eval()

        # Initialize variables for loss and accuracy
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_targets = []
        all_outputs=[]

        # Turn off gradient computation
        with torch.no_grad():
            for data in test_loader:
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
                test_loss += loss.item()

                # Compute accuracies
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)
                
                # Store predictions and targets for confusion matrix
                all_outputs.extend(outputs.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate validation accuracy and loss
        test_accuracy = correct_predictions / total_predictions
        avg_test_loss = test_loss / len(test_loader)

        return avg_test_loss, test_accuracy, all_predictions, all_targets

    
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


    def extract_pooler_output(self, data_loader, device):
            """
            Extracts the values present in output1["pooler_output"] and saves them in a new pandas DataFrame.

            Args:
                data_loader (DataLoader): DataLoader for the data.
                device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').

            Returns:
                pandas.DataFrame: DataFrame containing the extracted values from output1["pooler_output"].
            """
            # Set the model to evaluation mode
            self.eval()

            # List to store the extracted values along with their indices
            extracted_values = []

            # Turn off gradient computation
            with torch.no_grad():
                for idx, data in enumerate(data_loader):
                    # Take inputs
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

                    # Compute pooler output
                    output_1 = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=True)
                    pooler_output = output_1["pooler_output"]
                    
                    # Convert pooler_output tensor to numpy array
                    pooler_output_np = pooler_output.cpu().numpy()
    
                    # Reshape pooler_output_np to have one row per sample in the batch
                    num_samples = pooler_output_np.shape[0]
                    pooler_output_np_reshaped = pooler_output_np.reshape(num_samples, -1)

                    # Add the pooler output values along with their indices to the list
                    for i in range(num_samples):
                        extracted_values.append({"index": idx * data_loader.batch_size + i, **{f"pooler_output_{j}": val for j, val in enumerate(pooler_output_np_reshaped[i])}})

            print(extracted_values)
            # Convert the list of extracted values into a pandas DataFrame
            df = pd.DataFrame(extracted_values)

            return df


    def save_model(self, path):
        """
        Saves the model state to a file.

        Args:
            path (str): Path to save the model state.
        """
        torch.save(self.state_dict(), path)



