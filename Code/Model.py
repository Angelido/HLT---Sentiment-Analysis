import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

#Set the device for PyTorch operations based on availability:
device = torch.device("cuda" if torch.cuda.is_available() 
                      else  "mps" if torch.backends.mps.is_available()
                      else "cpu"
                      )