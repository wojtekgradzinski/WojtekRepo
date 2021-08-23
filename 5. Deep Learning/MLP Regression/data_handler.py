import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split



def load_data(pth, batch_size):

    #load data
    data = pd.read_csv(pth)
    
    
    data.drop(['date'], inplace = True)
    
    y = data['DAX'].values
    x = data.drop['DAX']
    
    
    
    
        