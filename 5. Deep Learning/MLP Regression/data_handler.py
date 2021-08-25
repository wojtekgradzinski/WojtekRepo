import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split



def load_data(pth, batch_size):

    #load data
    data = pd.read_csv(pth)
    
    #dropping suless columns
    data.drop(['date'], inplace = True, axis =1)
    
    #defining x,y
    y = data['DAX'].values
    x = data.drop(['DAX'], axis =1).values
    
    
    x, y = to_batches(x, y, batch_size)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 0)
    
    #speed up the algo  
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #putting data to tensor
    x_train = torch.tensor(x_train.astype(np.float32)).to(device)
    x_test = torch.tensor(x_test.astype(np.float32)).to(device)
    
    y_train = torch.tensor(y_train.astype(np.float32)).to(device)
    y_test = torch.tensor(y_test.astype(np.float32)).to(device)

    return x_train, x_train, y_test, y_train

def to_batches(x,y,batch_size):
    
    #defining number of batches
    n_batches = x.shape[0] // batch_size
    
     
    #random selection
    indexes = np.random.permutation(x.shape[0])
    
    x = x[indexes]
    y = y[indexes]
    
    #splitting data
    x = x[ : batch_size * n_batches ].reshape(n_batches, batch_size, x.shape[1])
    y = y[ : batch_size * n_batches ].reshape(n_batches, batch_size, 1)
    
    return x,y   

x_train, x_test, y_train, y_test = load_data('project/turkish_stocks.csv',4)    

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
        